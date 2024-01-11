import torch
import torch.nn as nn
import sys
sys.path.append("C:\\Users\\bunna\\Documents\\GitHub\\NeuralGraphPaper\\ES")
from ES_classes import ES_Linear, ES_Mult_Div, ES_MLP, ES_Easy_Zero, rank_normalize

class NeuralGraph(nn.Module):# connections = inputs to hid and action, hid to action, reward to everything.  Default const_n to label nodes as inp, action, hid and reward
    def __init__(self, n_hid, n_inputs, n_actions, ch_n=12, ch_e=8, ch_k=8, ch_inp=1, ch_n_const=4, ch_e_const=0,
                decay=0, clamp_mode="hard", max_value=100, init_mode="random", init_std=.05, set_nodes=False, 
                aggregation="mean", n_heads=1, n_models=1, messages=None, updates=None, attentions=None, 
                inp_enc=None, action_enc=None, reward_enc=None, action_dec=None, zero_ps=False):
        super(NeuralGraph, self).__init__()
        # FORGOT TO CONNECT HID TO ITSELF... JUST DID IT
        self.n_hid, self.n_inputs, self.n_actions = n_hid, n_inputs, n_actions
        self.n_nodes = n_inputs + n_hid + n_actions + 1 # 1 + for 1 reward node
        connections = [[i, j] for i in range(n_inputs) for j in range(n_inputs, n_inputs + n_hid + n_actions)] # input to hid and actions
        connections = connections + [[i, j] for i in range(n_inputs, n_inputs + n_hid) for j in range(n_inputs, n_inputs + n_hid + n_actions)] # hid to hid and actions
        connections = connections + [[self.n_nodes - 1, i] for i in range(self.n_nodes - 1)] # Reward to everything else (reward is last node)
        connections, self.n_edges = torch.Tensor(connections).long(), len(connections)
        
        self.connections = connections

        self.ch_n, self.ch_e, self.ch_k, self.ch_inp, self.ch_n_const, self.ch_e_const = ch_n, ch_e, ch_k, \
            ch_inp, ch_n_const, ch_e_const
        self.decay, self.n_models = decay, n_models
        self.clamp_mode, self.max_value = clamp_mode.lower(), max_value
        self.init_mode, self.set_nodes, self.aggregation = init_mode.lower(), set_nodes, aggregation.lower()
        self.init_std, self.n_heads = init_std, n_heads

        self.train_flag = True

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
            ES_Linear(2 + ch_n*2+ch_e, 32), # Getting deg info now too    
            nn.ReLU(),
            ES_Easy_Zero(32, 2*(ch_n-ch_n_const)+(ch_e-ch_e_const)),
        )) for _ in range(self.n_models)])

        self.updates = updates or nn.ModuleList([ES_MLP(nn.Sequential(
            ES_Linear(ch_n*3-ch_n_const*2, 16),
            nn.ReLU(),
            ES_Linear(16, ch_n-ch_n_const),
        )) for _ in range(self.n_models)])
        
        self.inp_enc = inp_enc or ES_MLP(nn.Sequential(
                ES_Linear(ch_n+ch_inp, 16),
                nn.ReLU(),
                ES_Linear(16, ch_n-ch_n_const),
            ))
        
        self.action_dec = action_dec or ES_MLP(nn.Sequential(
                ES_Linear(ch_n, 8),
                nn.ReLU(),
                ES_Linear(8, 1),
            ))
        
        self.action_enc = action_enc or ES_MLP(nn.Sequential(
                ES_Linear(ch_n+ch_inp, 16),
                nn.ReLU(),
                ES_Linear(16, ch_n-ch_n_const),
            ))
        
        self.reward_enc = reward_enc or ES_MLP(nn.Sequential(
                ES_Linear(ch_n+1, 16),
                nn.ReLU(),
                ES_Linear(16, ch_n-ch_n_const),
            ))


        self.ruleset = [self.messages, self.updates, self.inp_enc, self.action_dec, self.action_enc, self.reward_enc]

        if self.aggregation == 'attention':
            if self.n_heads > 1:
                self.multi_head_outa = ES_Linear(self.ch_n, self.ch_n, bias=False)
                self.multi_head_outb = ES_Linear(self.ch_n, self.ch_n, bias=False)
                self.ruleset.append(self.multi_head_outa)
                self.ruleset.append(self.multi_head_outb)

            self.attentions = attentions or nn.ModuleList([ES_MLP(nn.Sequential(
                ES_Linear(ch_n, 4*ch_k),
            )) for _ in range(self.n_models)])

            self.ruleset.append(self.attentions)

        self.register_buffer("const_n", torch.zeros(self.n_nodes, self.ch_n_const))
        self.register_buffer("const_e", torch.zeros(self.n_edges, self.ch_e_const))

        self.register_buffer("sources", connections[:, 0].long())
        self.register_buffer("targets", connections[:, 1].long())

        out_degs = torch.zeros(self.n_nodes).long().to(connections.device)
        in_degs = torch.zeros(self.n_nodes).long().to(connections.device)
        out_degs.index_add_(0, *torch.unique(self.sources, return_counts=True))
        in_degs.index_add_(0, *torch.unique(self.targets, return_counts=True))
        # To avoid div by 0
        # out_degs[out_degs == 0] = 1
        # in_degs[in_degs == 0] = 1
        # Commented out since I want to give this info to the message fn.
        # Will just fix this when dividing

        self.register_buffer("in_degs", in_degs)
        self.register_buffer("out_degs", out_degs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set default const_n values
        const_n = torch.zeros(self.n_nodes, ch_n_const).to(self.device)
        const_n[:n_inputs, 0] = 1 # Inputs
        const_n[n_inputs:n_inputs + n_hid, 1] = 1 # Hidden
        const_n[n_inputs + n_hid:n_inputs + n_hid + n_actions, 2] = 1 # Actions
        const_n[-1, 3] = 1 # Reward
        self.set_const_vals(const_n=const_n)

        if zero_ps:
            with torch.no_grad():
                for p in self.parameters():
                    p.fill_(0)
    
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

        # Get messages
        m_x = torch.cat([self.nodes[:, self.sources], self.nodes[:, self.targets], self.edges, \
                        self.out_degs[self.sources].unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1), \
                        self.in_degs[self.targets].unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)], dim=2)
        m = self.messages[t % self.n_models](m_x)
        
        m_a, m_b, m_ab = torch.tensor_split(m, [self.ch_n-self.ch_n_const, (self.ch_n-self.ch_n_const)*2], 2)
        
        if self.aggregation == "attention":
            m_a, m_b = self.calc_attention(m_a, m_b, t)

        # Aggregate messages
        agg_m_a = torch.zeros(batch_size, self.n_nodes, self.ch_n-self.ch_n_const).to(self.device)
        agg_m_b = torch.zeros(batch_size, self.n_nodes, self.ch_n-self.ch_n_const).to(self.device)
        agg_m_a.index_add_(1, self.sources, m_a)
        agg_m_b.index_add_(1, self.targets, m_b)

        if self.aggregation == "mean":
            agg_m_a.divide_(torch.where(self.out_degs != 0, self.out_degs, 1)[None, :, None])
            agg_m_b.divide_(torch.where(self.in_degs != 0, self.in_degs, 1)[None, :, None])
        
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
                self.edges[:, :, :self.ch_e-self.ch_e_const] = ((self.edges[:, :, :self.ch_e-self.ch_e_const] + m_ab*dt)/self.max_value).tanh() * self.max_value
            if self.clamp_mode == "hard":
                self.edges[:, :, :self.ch_e-self.ch_e_const] = (self.edges[:, :, :self.ch_e-self.ch_e_const] + m_ab*dt).clamp(-self.max_value, self.max_value)
            else:
                self.edges[:, :, :self.ch_e-self.ch_e_const] = self.edges[:, :, :self.ch_e-self.ch_e_const] + m_ab*dt

    # Refactored this bc it clutters the timestep function and I don't usually have attention turned on anyway currently
    def calc_attention(self, m_a, m_b, t):
        attention = self.attentions[t % self.n_models](self.nodes)
        attention = attention.reshape(*attention.shape[:-1], self.n_heads, (self.ch_k*4)//self.n_heads)
        
        f_keys, f_queries, b_keys, b_queries = torch.split(attention, [self.ch_k//self.n_heads,]*4, -1)
        f = (f_keys[:, self.sources] * f_queries[:, self.targets]).sum(-1)
        b = (b_queries[:, self.sources] * b_keys[:, self.targets]).sum(-1)

        # Very hard to get max trick working with this implementation of softmax
        # Therefore we will just do a soft clamp with tanh to ensure no value gets too big
        f_ws = torch.exp(10*torch.tanh(f/10))
        b_ws = torch.exp(10*torch.tanh(b/10))

        f_w_agg = torch.zeros(self.nodes.shape[0], self.n_nodes, self.n_heads).to(self.device)
        b_w_agg = torch.zeros(self.nodes.shape[0], self.n_nodes, self.n_heads).to(self.device)

        f_w_agg.index_add_(1, self.targets, f_ws)
        b_w_agg.index_add_(1, self.sources, b_ws)

        # batch, n_edges, n_heads
        f_attention = f_ws / f_w_agg[:, self.targets]
        b_attention = b_ws / b_w_agg[:, self.sources]

        # -> batch, n_edges, ch_n
        heads_a = m_a.reshape(*m_a.shape[:-1], self.n_heads, (self.ch_n-self.ch_n_const)//self.n_heads) * torch.repeat_interleave(b_attention.unsqueeze(-1), (self.ch_n-self.ch_n_const)//self.n_heads, -1)
        heads_b = m_b.reshape(*m_b.shape[:-1], self.n_heads, (self.ch_n-self.ch_n_const)//self.n_heads) * torch.repeat_interleave(f_attention.unsqueeze(-1), (self.ch_n-self.ch_n_const)//self.n_heads, -1)
        if self.n_heads > 1:
            m_a = self.multi_head_outa(heads_a.reshape(*m_a.shape))
            m_b = self.multi_head_outb(heads_b.reshape(*m_b.shape))
        else:
            m_a = heads_a.reshape(*m_a.shape)
            m_b = heads_b.reshape(*m_b.shape)

        return m_a, m_b

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
                if self.train_flag:
                    self.nodes = torch.cat([self.new_init_nodes, const_n], axis=2)
                else:
                    self.nodes = torch.cat([self.init_nodes.unsqueeze(0).repeat(batch_size, 1, 1), const_n], axis=2)
            elif self.init_mode == "random":
                self.nodes = torch.cat([torch.randn(batch_size, self.n_nodes, self.ch_n-self.ch_n_const).to(self.device) * self.init_std, const_n], axis=2)
            elif self.init_mode == "zeros":
                self.nodes = torch.cat([torch.zeros(batch_size, self.n_nodes, self.ch_n-self.ch_n_const).to(self.device), const_n], axis=2)
            else:
                raise RuntimeError(f"Unknown initial value config {self.init_mode}")
            
        if edges:
            const_e = torch.repeat_interleave(self.const_e.unsqueeze(0), batch_size, 0)
            if self.init_mode == "trainable":
                if self.train_flag:
                    self.edges = torch.cat([self.new_init_edges, const_e], axis=2)
                else:
                    self.edges = torch.cat([self.init_edges.unsqueeze(0).repeat(batch_size, 1, 1), const_e], axis=2)
            elif self.init_mode == "random":
                self.edges = torch.cat([torch.randn(batch_size, self.n_edges, self.ch_e-self.ch_e_const).to(self.device) * self.init_std, const_e], axis=2)
            elif self.init_mode == "zeros":
                self.edges = torch.cat([torch.zeros(batch_size, self.n_edges, self.ch_e-self.ch_e_const).to(self.device), const_e], axis=2)
            else:
                raise RuntimeError(f"Unknown initial value config {self.init_mode}")
    
    def apply_inputs(self, inp, dt=1):
        """
        Apply inputs to input nodes in the graph and labels to output nodes in the graph
        :param inp: The inputs for the graph.  Must be shape (batch_size, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_inputs)
        :param label: The labels for the graph.  Must be shape (batch_size, n_outputs, ch_out) unless ch_out == 1
            in which case it can just be (batch_size, n_outputs)
        """
        if self.ch_inp == 1 and len(inp.shape) != 3:
            inp = inp.unsqueeze(-1)

        enc = self.inp_enc(torch.cat([inp, self.nodes[:, :self.n_inputs]], axis=2))
        
        if self.set_nodes:
            self.nodes[:, :self.n_inputs, :self.ch_n-self.ch_n_const] = enc
        else:
            self.nodes[:, :self.n_inputs, :self.ch_n-self.ch_n_const] += enc*dt
    
    def get_action_dist(self):
        return self.action_dec(self.nodes[:, -self.n_actions-1:-1]).squeeze(-1)# .softmax(-1) DONT SOFTMAX FOR CE

    def apply_feedback(self, action, reward, dt=1):
        action_oh = nn.functional.one_hot(action, self.n_actions).unsqueeze(-1)
        reward = reward.reshape(-1, 1, 1)

        action_enc = self.action_enc(torch.cat([action_oh, self.nodes[:, -self.n_actions-1:-1]], axis=2))
        reward_enc = self.reward_enc(torch.cat([reward, self.nodes[:, -1:]], axis=2))

        if self.set_nodes:
            self.nodes[:, -self.n_actions-1:-1, :self.ch_n-self.ch_n_const] = action_enc
            self.nodes[:, -1:, :self.ch_n-self.ch_n_const] = reward_enc
        else:
            self.nodes[:, -self.n_actions-1:-1, :self.ch_n-self.ch_n_const] += action_enc*dt
            self.nodes[:, -1:, :self.ch_n-self.ch_n_const] += reward_enc*dt


    def overflow(self, k=5):
        """
        Calculates the overflow of the graph intended to be added to loss to prevent explosions.
        :param k: maximum value of node and edge values before they incur a penalty.
        """
        node_overflow = ((self.nodes - self.nodes.clamp(-k, k))**2).mean()
        edge_overflow = ((self.edges - self.edges.clamp(-k, k))**2).mean()
        return node_overflow + edge_overflow
        
    def forward(self, inp, time=5, dt=1, apply_once=False, nodes=True, edges=False, **kwargs):
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
            self.apply_inputs(inp)# , dt=dt

        for t in range(timesteps):
            if not apply_once:
                self.apply_inputs(inp, dt=dt)
            self.timestep(step_nodes=nodes, step_edges=edges, dt=dt, t=t)
        return self.get_action_dist()
    
    def backward(self, inp, action, reward, time=5, dt=1, apply_once=False, nodes=True, edges=False, edges_at_end=True, **kwargs):
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
            self.apply_inputs(inp)
            self.apply_feedback(action, reward)

        for t in range(timesteps-1):
            if not apply_once:
                self.apply_inputs(inp, dt=dt)
                self.apply_feedback(action, reward, dt=dt)
            self.timestep(step_nodes=nodes, step_edges=edges, dt=dt, t=t)

        self.timestep(step_nodes=nodes, step_edges=edges or edges_at_end, dt=dt, t=t)

    def generate_epsilons(self, batch_size, sigma=.1):
        if self.init_mode == "trainable":
            init_nodes_eps = torch.randn(batch_size//2, *self.init_nodes.shape).to(self.device) * sigma
            init_edges_eps = torch.randn(batch_size//2, *self.init_edges.shape).to(self.device) * sigma

            self.init_nodes_eps = torch.cat([init_nodes_eps, -init_nodes_eps], axis=0)
            self.init_edges_eps = torch.cat([init_edges_eps, -init_edges_eps], axis=0)

            self.new_init_nodes = self.init_nodes.unsqueeze(0).repeat(batch_size, 1, 1) + self.init_nodes_eps
            self.new_init_edges = self.init_edges.unsqueeze(0).repeat(batch_size, 1, 1) + self.init_edges_eps
            

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
        
        self.train_flag = flag

    def estimate_grads(self, losses, sigma=.1, normalize=True):
        if normalize:
            losses = rank_normalize(losses)
        
        if self.init_mode == "trainable":
            # Estimate gradients
            self.init_nodes.grad = (self.init_nodes_eps * losses.reshape(-1, 1, 1)).mean(0) / sigma**2
            self.init_edges.grad = (self.init_edges_eps * losses.reshape(-1, 1, 1)).mean(0) / sigma**2

        for model in self.ruleset:
            if isinstance(model, nn.ModuleList):
                for sub_model in model:
                    sub_model.estimate_grads(losses, sigma=sigma, normalize=False)# Already would've been normalized loss
            else:
                model.estimate_grads(losses, sigma=sigma, normalize=False)

    def to(self, device):
        self.device = device
        self.const_n = self.const_n.to(device)
        return super(NeuralGraph, self).to(device)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save_ruleset(self, path):
        for i, rule in enumerate(self.ruleset):
            torch.save(rule.state_dict(), path.replace(".pt", f"_{i}.pt"))
    
    def load_ruleset(self, path):
        for i, rule in enumerate(self.ruleset):
            rule.load_state_dict(torch.load(path.replace(".pt", f"_{i}.pt")))