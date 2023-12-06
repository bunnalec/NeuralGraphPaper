import torch
import torch.nn as nn
from .ES_classes import ES_Linear, ES_MLP

class NeuralGraph(nn.Module):
    def __init__(self, n_nodes, n_inputs, n_outputs, connections, ch_n=8, ch_e=8, ch_k=8, ch_inp=1, ch_out=1, ch_n_const=0, ch_e_const=0,
                decay=0, clamp_mode="soft", max_value=1e6, init_mode="trainable", init_std=1, set_nodes=False, 
                aggregation="attention", n_heads=1, use_label=False,
                n_models=1, messages=None, updates=None, attentions=None, 
                inp_enc=None, label_enc=None, out_dec=None):
        """
        Creates a Neural Graph.  A Neural Graph is an arbitrary directed graph which has input nodes and output nodes.  
        Every node and edge in the graph has a state which is represented as a vector with dimensionality of ch_n and ch_e repectively.
        Information flows through the graph according to a set of rules determined by 2-3 functions.
        Each timestep every triplet of (node_a, node_b, edge_ab) calculates three "messages" (m_a, m_b, m_ab) according to a message function.
        The edge message m_ab can be simply added to the edge value, however each node might have many messages from numerous connections.
        Therefore, these messages are aggregated using sum (or an attention function) and then passed to the update function, which will
        take the aggregated messages and produce the node update which is added to the node values.  This is the end of one timestep.


        :param n_nodes: Total number of nodes in the graph
        :param n_inputs: Number of input nodes in the graph (indices 0:n_inputs)
        :param n_outputs: Number of output nodes in the graph (indices -n_ouputs:-1)
        :param connections: List of connections of nodes e.g. [(0, 1), (1, 2), ... (0, 2)]
        
        :param ch_n: Number of channels for each node
        :param ch_e: Number of channels for each edge
        :param ch_k: Number of channels for attention keys/queries
        :param ch_inp: Number of input channels
        :param ch_out: Number of output channels
        :param ch_n_const: Number of constant channels in nodes
        :param ch_e_const: Number of constant channels in edges
        
        :param decay: Amount node values are decayed each unit of time
        :param leakage: Given a node A, the average values of all nodes connected to A 
            (in either direction) will be averaged and put into A at each timestep.
            It will be combined with A's original value according to the equation
            nodeA = (1-leakage) * nodeA + leakage * avg_connect_node
        :param clamp_mode: One of [soft, hard, none]
        :param max_value: The maximum (absolute) value that node and edge values can take. Anything larger
            than this value will be clamped to be below
        :param value_init: One of [trainable, trainable_batch, random, zeros].  Decides how to initialize node and edges.  
            If using trainable_batch make sure to call graph.init_vals(batch_size=___) BEFORE making an optimizer with graph.parameters().
            Otherwise the initial values will not be registered or trained.
        :param init_value_std: Standard deviation of initial values
        :param set_nodes: Instead of adding to a node's current value every timestep, it will set the node.
        :param aggregation: One of [attention, sum, mean].  How to aggregate messages
        :param device: What device to run everything on
        :param use_label: Whether to be able to input the label to an example into the output nodes of the graph for pseudolearning
        :param node_dropout_p: What percent of node updates to drop to 0
        :param edge_dropout_p: What percent of edge updates to drop to 0
        :param poolsize: Poolsize to use for persistence training (if set to None then no persistence training)

        :param n_models: Number of models to cycle through
        :param message_generator: Function to generate message models.  Must have very 
            specific shape of input_shape=(ch_n*2 + ch_e) and output_shape=(ch_n*2 + ch_e)
        :param update_generator: Function to generate update models.  Must have very 
            specific shape of input_shape=(ch_n*3) and output_shape=(ch_n)
        :param attention_generator: Function to generate attention models.  Must have very 
            specific shape of input_shape=(ch_n) and output_shape=(ch_k*4)
        :param inp_int_generator: Function to generate input integrator.  Must have very
            specific shape of input_shape=(ch_inp+ch_n) and output_shape=(ch_n)
        :param label_int_generator: Function to generate label integrator.  Must have very
            specific shape of input_shape=(ch_out+ch_n) and output_shape=(ch_n)
        :param out_int_generator: Function to generate output interpreter.  Must have very
            specific shape of input_shape=(ch_n) and output_shape=(ch_out)
        """
        super(NeuralGraph, self).__init__()
        
        self.n_nodes, self.n_edges, self.connections = n_nodes, len(connections), connections
        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        self.ch_n, self.ch_e, self.ch_k, self.ch_inp, self.ch_out, self.ch_n_const, self.ch_e_const = ch_n, ch_e, ch_k, \
            ch_inp, ch_out, ch_n_const, ch_e_const
        self.decay, self.n_models = decay, n_models
        self.clamp_mode, self.max_value = clamp_mode.lower(), max_value
        self.init_mode, self.set_nodes, self.aggregation, self.use_label = init_mode, set_nodes, aggregation, use_label
        self.init_std, self.n_heads = init_std, n_heads

        assert self.clamp_mode in ["soft", "hard", "none"], f"Unknown clamp_mode option {self.clamp_mode}"
        assert self.aggregation in ["attention", "sum", "mean"], f"Unknown aggregation option {self.aggregation}"
        assert self.init_mode in ["trainable", "random", "zeros"], f"Unknown initialization option {self.init_mode}"
        assert (self.ch_n-self.ch_n_const) % self.n_heads == 0 and self.ch_k % self.n_heads == 0, "ch_n and ch_k need to be divisible by n_heads"
        
        if self.init_mode == "trainable":
            self.register_parameter("init_nodes", nn.Parameter(torch.randn(self.n_nodes, ch_n-ch_n_const) * self.init_std, requires_grad=True))
            self.register_parameter("init_edges", nn.Parameter(torch.randn(self.n_edges, ch_e-ch_e_const) * self.init_std, requires_grad=True))

        self.register_buffer('nodes', torch.zeros(1, self.n_nodes, self.ch_n), persistent=False)
        self.register_buffer('edges', torch.zeros(1, self.n_edges, self.ch_e), persistent=False)

        # Default message
        self.messages = messages or nn.ModuleList([ES_MLP(nn.Sequential(
            ES_Linear(ch_n*2+ch_e, 16),
            nn.Tanh(),
            ES_Linear(16, 2*(ch_n-ch_n_const)+(ch_e-ch_e_const)),
        )) for _ in range(self.n_models)])


        self.updates = updates or nn.ModuleList([ES_MLP(nn.Sequential(
            ES_Linear(ch_n*3, 16),
            nn.Tanh(),
            ES_Linear(16, ch_n-ch_n_const),
        )) for _ in range(self.n_models)])
        
        self.inp_enc = inp_enc or ES_MLP(nn.Sequential(
                ES_Linear(ch_n+ch_inp, 16),
                nn.Tanh(),
                ES_Linear(16, ch_n-ch_n_const),
            ))
        self.out_dec = out_dec or ES_MLP(nn.Sequential(
                ES_Linear(ch_n, 16),
                nn.Tanh(),
                ES_Linear(16, ch_out),
            ))

        self.ruleset = [self.messages, self.updates, self.inp_enc, self.out_dec]

        if self.aggregation == 'attention':
            if self.n_heads > 1:
                self.multi_head_outa = ES_Linear(self.ch_n, self.ch_n, bias=False)
                self.multi_head_outb = ES_Linear(self.ch_n, self.ch_n, bias=False)
                self.ruleset.append(self.multi_head_outa)
                self.ruleset.append(self.multi_head_outb)

            self.attentions = attentions or nn.ModuleList([ES_MLP(nn.Sequential(
                ES_Linear(ch_n, 16),
                nn.Tanh(),
                ES_Linear(16, 4*ch_k),
            )) for _ in range(self.n_models)])

            self.ruleset.append(self.attentions)
        
        if self.use_label:
            self.label_enc = label_enc or ES_MLP(nn.Sequential(
                ES_Linear(ch_n+ch_out, 16),
                nn.Tanh(),
                ES_Linear(16, ch_n-ch_n_const),
            ))

            self.ruleset.append(self.label_enc)

        self.register_buffer("const_n", torch.zeros(self.n_nodes, self.ch_n_const))
        self.register_buffer("const_e", torch.zeros(self.n_edges, self.ch_n_const))

        self.register_buffer("sources", connections[:, 0].long())
        self.register_buffer("targets", connections[:, 1].long())

        out_degs = torch.zeros(self.n_nodes).long()
        in_degs = torch.zeros(self.n_nodes).long()
        out_degs.index_add_(0, *torch.unique(self.sources, return_counts=True))
        in_degs.index_add_(0, *torch.unique(self.targets, return_counts=True))
        # To avoid div by 0
        out_degs[out_degs == 0] = 1
        in_degs[in_degs == 0] = 1

        self.register_buffer("in_degs", in_degs)
        self.register_buffer("out_degs", out_degs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def set_const_vals(self, const_n=None, const_e=None):
        assert const_n is None or const_n.shape == (self.n_nodes, self.ch_n_const)
        assert const_e is None or const_e.shape == (self.n_edges, self.ch_e_const)
        if not const_n is None:
            self.const_n = const_n
        if not const_e is None:
            self.const_e = const_e

    def timestep(self, dt=1, step_nodes=True, step_edges=True, t=0):
        """
        Runs one timestep of the Neural Graph.
        :param dt: The temporal resolution of the timestep.  At the limit of dt->0, the rules become differential equations.
            This was inspired by smoothlife/lenia
        :param step_nodes: Whether to update node values this timestep
        :param step_edges: Whether to update edge values this timestep
        :param t: The current timestep (used to decide which set of models to use)
        """

        batch_size = self.nodes.shape[0]

        # nodes = self.nodes.clone() if (self.pool is None) else self.nodes[self.pool]
        # edges = self.edges.clone() if (self.pool is None) else self.edges[self.pool]

        # Get messages
        m_x = torch.cat([self.nodes[:, self.sources], self.nodes[:, self.targets], self.edges], dim=2)
        m = self.messages[t % self.n_models](m_x)
        
        m_a, m_b, m_ab = torch.tensor_split(m, [self.ch_n-self.ch_n_const, (self.ch_n-self.ch_n_const)*2], 2)
        
        if self.aggregation == "attention":
            attention = self.attentions[t % self.n_models](self.nodes)
            attention = attention.reshape(*attention.shape[:-1], self.n_heads, (self.ch_k*4)//self.n_heads)
            
            f_keys, f_queries, b_keys, b_queries = torch.split(attention, [self.ch_k//self.n_heads,]*4, -1)
            f = (f_keys[:, self.sources] * f_queries[:, self.targets]).sum(-1)
            b = (b_queries[:, self.sources] * b_keys[:, self.targets]).sum(-1)

            # Very hard to get max trick working with this implementation of softmax
            # Therefore we will just do a soft clamp with tanh to ensure no value gets too big
            f_ws = torch.exp(10*torch.tanh(f/10))
            b_ws = torch.exp(10*torch.tanh(b/10))

            f_w_agg = torch.zeros(batch_size, self.n_nodes, self.n_heads).to(self.device)
            b_w_agg = torch.zeros(batch_size, self.n_nodes, self.n_heads).to(self.device)

            f_w_agg.index_add_(1, self.targets, f_ws)
            b_w_agg.index_add_(1, self.sources, b_ws)

            # batch, n_edges, n_heads
            f_attention = f_ws / f_w_agg[:, self.targets]
            b_attention = b_ws / b_w_agg[:, self.sources]

            # -> batch, n_edges, ch_n
            heads_b = m_b.reshape(*m_b.shape[:-1], self.n_heads, (self.ch_n-self.ch_n_const)//self.n_heads) * torch.repeat_interleave(f_attention.unsqueeze(-1), (self.ch_n-self.ch_n_const)//self.n_heads, -1)
            heads_a = m_a.reshape(*m_a.shape[:-1], self.n_heads, (self.ch_n-self.ch_n_const)//self.n_heads) * torch.repeat_interleave(b_attention.unsqueeze(-1), (self.ch_n-self.ch_n_const)//self.n_heads, -1)
            if self.n_heads > 1:
                m_b = self.multi_head_outb(heads_b.reshape(*m_b.shape))
                m_a = self.multi_head_outa(heads_a.reshape(*m_a.shape))
            else:
                m_b = heads_b.reshape(*m_b.shape)
                m_a = heads_a.reshape(*m_a.shape)

        # Aggregate messages
        agg_m_a = torch.zeros(batch_size, self.n_nodes, self.ch_n-self.ch_n_const).to(self.device)
        agg_m_b = torch.zeros(batch_size, self.n_nodes, self.ch_n-self.ch_n_const).to(self.device)
        agg_m_a.index_add_(1, self.sources, m_a)
        agg_m_b.index_add_(1, self.targets, m_b)

        if self.aggregation == "mean":
            agg_m_a.divide_(self.out_degs[None, :, None])
            agg_m_b.divide_(self.in_degs[None, :, None])
        
        # Get updates
        u_x = torch.cat([agg_m_a, agg_m_b, self.nodes], axis=2)
        update = self.updates[t % self.n_models](u_x)

        if step_nodes:
            if self.set_nodes:
                _nodes = update
            else:
                _nodes = self.nodes[:, :, :self.ch_n-self.ch_n_const]
                if self.decay > 0:
                    _nodes = _nodes * ((1-self.decay)**dt)
                _nodes = _nodes + update*dt

            if self.clamp_mode == "soft":
                _nodes = (_nodes/self.max_value).tanh() * self.max_value
            elif self.clamp_mode == "hard":
                _nodes = _nodes.clamp(-self.max_value, self.max_value)

            self.nodes[:, :, :self.ch_n-self.ch_n_const] = _nodes

        if step_edges:
            if self.clamp_mode == "soft":
                self.edges[:, :, :self.ch_e-self.ch_e_const] = ((self.edges[:, :, :self.ch_e-self.ch_e_const] + m_ab)/self.max_value).tanh() * self.max_value
            if self.clamp_mode == "hard":
                self.edges[:, :, :self.ch_e-self.ch_e_const] = (self.edges[:, :, :self.ch_e-self.ch_e_const] + m_ab).clamp(-self.max_value, self.max_value)
            else:
                self.edges[:, :, :self.ch_e-self.ch_e_const] = self.edges[:, :, :self.ch_e-self.ch_e_const] + m_ab

    
    def init_vals(self, batch_size, nodes=True, edges=True):
        """
        Initialize nodes and edges.
        :param nodes: Whether to initialize nodes
        :param edges: Whether to initialize edeges
        :batch_size: Batch size of the initialization
        """
        if nodes:
            const_n = torch.repeat_interleave(self.const_n.unsqueeze(0), batch_size, 0)
            if self.init_mode == "trainable":
                self.nodes = torch.cat([torch.repeat_interleave((self.init_nodes).clone().unsqueeze(0), batch_size, 0), const_n], axis=2)
            elif self.init_mode == "random":
                self.nodes = torch.cat([torch.randn(batch_size, self.n_nodes, self.ch_n-self.ch_n_const).to(self.device) * self.init_std, const_n], axis=2)
            elif self.init_mode == "zeros":
                self.nodes = torch.cat([torch.zeros(batch_size, self.n_nodes, self.ch_n-self.ch_n_const).to(self.device), const_n], axis=2)
            else:
                raise RuntimeError(f"Unknown initial value config {self.init_mode}")
            
        if edges:
            const_e = torch.repeat_interleave(self.const_e.unsqueeze(0), batch_size, 0)
            if self.init_mode == "trainable":
                self.edges = torch.cat([torch.repeat_interleave((self.init_edges).clone().unsqueeze(0), batch_size, 0), const_e], axis=2)
            elif self.init_mode == "random":
                self.edges = torch.cat([torch.randn(batch_size, self.n_edges, self.ch_e-self.ch_e_const).to(self.device) * self.init_std, const_e], axis=2)
            elif self.init_mode == "zeros":
                self.edges = torch.cat([torch.zeros(batch_size, self.n_edges, self.ch_e-self.ch_e_const).to(self.device), const_e], axis=2)
            else:
                raise RuntimeError(f"Unknown initial value config {self.init_mode}")
    
    def apply_vals(self, inp=None, label=None):
        """
        Apply inputs to input nodes in the graph and labels to output nodes in the graph
        :param inp: The inputs for the graph.  Must be shape (batch_size, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_inputs)
        :param label: The labels for the graph.  Must be shape (batch_size, n_outputs, ch_out) unless ch_out == 1
            in which case it can just be (batch_size, n_outputs)
        """
        if inp is not None:
            if self.ch_inp == 1 and len(inp.shape) != 3:
                inp = inp.unsqueeze(-1)

            enc = self.inp_enc(torch.cat([inp, self.nodes[:, :self.n_inputs]], axis=2))
            
            if self.set_nodes:
                self.nodes[:, :self.n_inputs, :self.ch_n-self.ch_n_const] = enc
            else:
                self.nodes[:, :self.n_inputs, :self.ch_n-self.ch_n_const] += enc
            
        if label is not None:
            assert self.use_label, "Tried to apply labels but use_label was set to False"
            if self.ch_out == 1 and len(label.shape) != 3:
                label = label.unsqueeze(-1)

            enc = self.label_enc(torch.cat([label, self.nodes[:, -self.n_outputs:]], axis=2))
            if self.set_nodes:
                self.nodes[:, -self.n_outputs:, :self.ch_n-self.ch_n_const] = enc
            else:
                self.nodes[:, -self.n_outputs:, :self.ch_n-self.ch_n_const] += enc
    
    def read_outputs(self):
        """
        Reads the outputs of the graph
        :return: outputs of shape (batch_size, n_outputs, ch_out) unless ch_out == 1 then (batch_size, n_outputs)
        """

        if self.ch_out == 1:
            return self.out_dec(self.nodes[:, -self.n_outputs:]).squeeze(-1)
        return self.out_dec(self.nodes[:, -self.n_outputs:])
    
    def overflow(self, k=5):
        """
        Calculates the overflow of the graph intended to be added to loss to prevent explosions.
        :param k: maximum value of node and edge values before they incur a penalty.
        """
        node_overflow = ((self.nodes - self.nodes.clamp(-k, k))**2).mean()
        edge_overflow = ((self.edges - self.edges.clamp(-k, k))**2).mean()
        return node_overflow + edge_overflow
        
    def forward(self, x, time=5, dt=1, apply_once=False, nodes=True, edges=True, **kwargs):
        """
        Take input x and run the graph for a certain amount of time with a certain time resolution.
        :param x: Input to the graph.  Must be shape (batch_size, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_inputs)
        :param time: time to run the graph for
        :param dt: Time resolution of the graph
        :param apply_once: Whether to apply the input once at the beginning or every timestep
        :param nodes: Whether to update nodes
        :param edges: Whether to update edges

        :return: The output of the graph after running on the inputs
        """
        timesteps = round(time / dt)

        if apply_once:
            self.apply_vals(x)

        for t in range(timesteps):
            if not apply_once:
                self.apply_vals(x)
            self.timestep(step_nodes=nodes, step_edges=edges, dt=dt, t=t)
        return self.read_outputs()
    
    def backward(self, x, y, time=5, dt=1, apply_once=False, nodes=True, edges=True, edges_at_end=True, **kwargs):
        """
        Takes an input x and a label y and puts them in the graph and runs the graph for a certain amount of time with a certain time resolution.
        :param x: Input for the graph.  Must be shape (batch_size, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_inputs)
        :param y: Label for the graph.  Must be shape (batch_size, n_outputs, ch_out) unless ch_out == 1
            in which case it can just be (batch_size, n_outputs)
        :param time: time to run the graph for
        :param dt: Time resolution of the graph
        :param apply_once: Whether to apply the input once at the beginning or every timestep
        :param nodes: Whether to update nodes every timestep
        :param edges: Whether to update edges every timestep
        :param edges_at_end: Whether to update edges on the last timestep
        """
        timesteps = round(time / dt)

        if apply_once:
            self.apply_vals(x, label=y)

        for t in range(timesteps-1):
            if not apply_once:
                self.apply_vals(x, label=y)
            self.timestep(step_nodes=nodes, step_edges=edges, dt=dt, t=t)

        self.timestep(step_nodes=nodes, step_edges=edges or edges_at_end, dt=dt, t=t)
    
    def predict(self, X, time=5, dt=1, reset_nodes=False, **kwargs):
        """
        Take a series of inputs and one by one inserts them into the graph and runs a forward pass.
        :param X:  Inputs for the graph.  Must be shape (batch_size, n_examples, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_examples, n_inputs)
        :param time: time to run the graph for
        :param dt: Time resolution of the graph
        :reset_nodes: Whether to reset the nodes after every forward pass

        :return: The outputs of the graph after the examples with shape (batch_size, n_examples, n_outputs, ch_out) unless ch_out == 1 
            in which case it is (batch_size, n_examples, n_outputs)
        """
        preds = []
        for i in range(X.shape[1]):
            if reset_nodes:
                self.init_vals(X.shape[0], nodes=True, edges=False)

            preds.append(self.forward(X[:, i], dt=dt, time=time, **kwargs))
        return torch.stack(preds, axis=1)
    
    def learn(self, X, Y, time=5, dt=1, reset_nodes=False, **kwargs):
        """
        Take a series of inputs and inserts them into the graph, runs a forward pass and then insert the correspond labels into the graph
        and runs a backwards pass.  
        :param X:  Inputs for the graph.  Must be shape (batch_size, n_examples, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_examples, n_inputs)
        :param Y:  Labels for the graph.  Must be shape (batch_size, n_examples, n_outputs, ch_out) unless ch_out == 1 
            in which case it can just be (batch_size, n_examples, n_outputs)
        :param time: time to run the graph for
        :param dt: Time resolution of the graph
        :reset_nodes: Whether to reset the nodes after every forward/backward pair
        """

        for i in range(X.shape[1]):
            if reset_nodes:
                self.init_vals(X.shape[0], nodes=True, edges=False)

            self.forward(X[:, i], dt=dt, time=time, **kwargs)
            self.backward(X[:, i], Y[:, i], dt=dt, time=time, **kwargs)

    def generate_epsilons(self, batch_size, sigma=.1):
        for model in self.ruleset:
            if isinstance(model, nn.ModuleList):
                for sub_model in model:
                    sub_model.generate_epsilons(batch_size, sigma=sigma)
            else:
                model.generate_epsilons(batch_size, sigma=sigma)

    def train_mode(self, flag):
        for model in self.ruleset:
            if isinstance(model, nn.ModuleList):
                for sub_model in model:
                    sub_model.train_mode(flag)
            else:
                model.train_mode(flag)

    def estimate_grads(self, losses, sigma=.1):
        for model in self.ruleset:
            if isinstance(model, nn.ModuleList):
                for sub_model in model:
                    sub_model.estimate_grads(losses, sigma=sigma)
            else:
                model.estimate_grads(losses, sigma=sigma)

    def to(self, device):
        self.device = device
        return super(NeuralGraph, self).to(device)
