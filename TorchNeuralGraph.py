from torch_geometric.nn import MessagePassing
from torch import nn
import torch
import numpy as np

class MLP(nn.Module):
    def __init__(self, ch_inp:int=8, ch_out:int=8):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(ch_inp, 32),
            nn.SiLU(),
            nn.Linear(32, ch_out),
        )
    
    def forward(self, x):
        return self.main(x)


class NeuralGraph(MessagePassing):
    def __init__(self, n_nodes, n_inputs, n_outputs, edge_index, ch_n=8, ch_e=8, ch_inp=1, ch_out=1,
        clamp_mode="soft", max_val=1e6, init_value="trainable", aggregation="add", init_std=1, use_label=False, 
        node_drop=0, edge_drop=0, device="cpu", n_models=1):

        super(NeuralGraph, self).__init__(aggr=aggregation)

        self.n_nodes, self.n_inputs, self.n_outputs = n_nodes, n_inputs, n_outputs
        self.edge_index = edge_index
        self.ch_n, self.ch_e, self.ch_inp, self.ch_out = ch_n, ch_e, ch_inp, ch_out
        self.clamp_mode, self.max_val, self.init_value, self.init_std = clamp_mode, max_val, init_value, init_std
        self.use_label = use_label
        self.node_drop, self.edge_drop = node_drop, edge_drop
        self.device = device
        self.n_models = n_models

        self.inp_enc = MLP(self.ch_inp+self.ch_n, self.ch_n).to(self.device)
        self.out_dec = MLP(self.ch_n, self.ch_out).to(self.device)
        self.label_enc = MLP(self.ch_out+self.ch_n, self.ch_n).to(self.device)

        self.f_messages = nn.ModuleList([MLP(2*self.ch_n+self.ch_e, self.ch_n)]).to(self.device)
        self.b_messages = nn.ModuleList([MLP(2*self.ch_n+self.ch_e, self.ch_n)]).to(self.device)
        self.node_updates = nn.ModuleList([MLP(3*self.ch_n, self.ch_n)]).to(self.device)
        self.edge_updates = nn.ModuleList([MLP(2*self.ch_n+self.ch_e, self.ch_e)]).to(self.device)

        self.n_edges = edge_index.shape[1]

        if self.init_value == "trainable":
            self.init_nodes = nn.Parameter(torch.randn(self.n_nodes, self.ch_n, device=self.device) * self.init_std)
            self.init_edges = nn.Parameter(torch.randn(self.n_edges, self.ch_e, device=self.device) * self.init_std)

        self.nodes = torch.zeros(1, self.n_nodes, self.ch_n, device=self.device)
        self.edges = torch.zeros(1, self.n_edges, self.ch_e, device=self.device)


    def message(self, x_i, x_j, edges, forward=True, t=0):
        if forward:
            # print(x_i.shape, x_j.shape, edges.shape)
            return self.f_messages[t % self.n_models](torch.cat([x_i, x_j, edges], axis=-1))
        else:
            return self.b_messages[t % self.n_models](torch.cat([x_i, x_j, edges], axis=-1))

    def edge_update(self, x_i, x_j, edges, t=0):
        return self.edge_updates[t % self.n_models](torch.cat([x_i, x_j, edges], axis=-1))

    def forward(self, nodes, edges, edge_index, t=0, dt=1):

        agg_f = self.propagate(edge_index, x=nodes, edges=edges, forward=True)
        agg_b = self.propagate(torch.flip(edge_index, dims=(0,)), x=nodes, edges=edges, forward=False)

        node_update = self.node_updates[t % self.n_models](torch.cat([agg_f, agg_b, nodes], axis=-1))
        edge_update = self.edge_updater(edge_index, x=nodes, edges=edges)

        return node_update * dt, edge_update
    
    def timestep(self, step_nodes=True, step_edges=True, t=0, dt=1):
        node_updates, edge_updates = self.forward(self.nodes, self.edges, self.edge_index, t=t, dt=dt)
        
        if step_nodes:
            self.nodes = self.nodes + node_updates
            

        if step_edges:
            self.edges = self.edges + edge_updates

    def clamp(self):
        if self.clamp_mode == "soft":
            self.nodes = self.max_val * nn.functional.tanh(self.nodes/self.max_val)
            self.edges = self.max_val * nn.functional.tanh(self.edges/self.max_val)
        
        if self.clamp_mode == "hard":
            self.nodes = self.nodes.clamp(-self.max_val, self.max_val)
            self.edges = self.edges.clamp(-self.max_val, self.max_val)

        if self.clamp_mode == "none":
            return
        
    def init_vals(self, nodes=True, edges=True, batch_size=1):
        """
        Initialize nodes and edges.
        :param nodes: Whether to initialize nodes
        :param edges: Whether to initialize edeges
        :batch_size: Batch size of the initialization
        """
        if nodes:
            if self.init_value == "trainable":
                self.nodes = torch.repeat_interleave((self.init_nodes).clone().unsqueeze(0), batch_size, 0)
            elif self.init_value == "random":
                self.nodes = torch.randn(batch_size, self.n_nodes, self.ch_n, device=self.device) * self.init_std
            elif self.init_value == "zeros":
                self.nodes = torch.zeros(batch_size, self.n_nodes, self.ch_n, device=self.device)
            else:
                raise RuntimeError(f"Unknown initial value config {self.init_value}")
            
        if edges:
            if self.init_value == "trainable":
                self.edges = torch.repeat_interleave((self.init_edges).clone().unsqueeze(0), batch_size, 0)
            elif self.init_value == "random":
                self.edges = torch.randn(batch_size, self.n_edges, self.ch_e, device=self.device) * self.init_std
            elif self.init_value == "zeros":
                self.edges = torch.zeros(batch_size, self.n_edges, self.ch_e, device=self.device)
            else:
                raise RuntimeError(f"Unknown initial value config {self.init_value}")

    def apply_vals(self, inp, label=None):
        """
        Apply inputs to input nodes in the graph and labels to output nodes in the graph
        :param inp: The inputs for the graph.  Must be shape (batch_size, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_inputs)
        :param label: The labels for the graph.  Must be shape (batch_size, n_outputs, ch_out) unless ch_out == 1
            in which case it can just be (batch_size, n_outputs)
        """

        if not type(inp) == torch.Tensor:
            inp = torch.tensor(inp, device=self.device)
        if self.ch_inp == 1 and len(inp.shape) != 3:
            inp = inp.unsqueeze(-1)

        int_inps = self.inp_enc(torch.cat([inp, self.nodes[:, :self.n_inputs]], axis=2))
        self.nodes[:, :self.n_inputs] += int_inps
        
        if label is not None:
            assert self.use_label, "Tried to apply labels but use_label was set to False"

            if not type(label) == torch.Tensor:
                label = torch.tensor(label, device=self.device)
            if self.ch_out == 1 and len(label.shape) != 3:
                label = label.unsqueeze(-1)
            

            int_label = self.label_enc(torch.cat([label, self.nodes[:, -self.n_outputs:]], axis=2))
            self.nodes[:, -self.n_outputs:] += int_label

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
    
    def forward_pass(self, x, time=5, dt=1, apply_once=False, nodes=True, edges=False):
        """
        Take input x and run the graph for a certain amount of time with a certain time resolution.
        :param x: Input to the graph.  Must be shape (batch_size, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_inputs)
        :param time: time to run the graph for
        :param dt: Time resolution of the graph
        :param apply_once: Whether to apply the input once at the beginning or every timestep
        :param nodes: Whether to update nodes
        :param edges: Whether to update edges
        """
        timesteps = round(time / dt)

        if apply_once:
            self.apply_vals(x)

        for t in range(timesteps):
            if not apply_once:
                self.apply_vals(x)
            self.timestep(step_nodes=nodes, step_edges=edges, dt=dt, t=t)
        return self.read_outputs()
    
    def backward_pass(self, x, y, time=5, dt=1, apply_once=False, nodes=True, edges=False, edges_at_end=True):
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
                self.init_vals(nodes=True, edges=False, batch_size=self.nodes.shape[0])

            preds.append(self.forward_pass(X[:, i], dt=dt, time=time, **kwargs))
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
                self.init_vals(nodes=True, edges=False, batch_size=self.nodes.shape[0])

            self.forward_pass(X[:, i], dt=dt, time=time, **kwargs)
            self.backward_pass(X[:, i], Y[:, i], dt=dt, time=time, **kwargs)
    
    def detach_vals(self):
        """
        Detach the node and edge values from the torch compute graph.
        """
        self.nodes = self.nodes.detach()
        self.edges = self.edges.detach()

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))