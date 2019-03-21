import torch
import torch.nn as nn

def _sru_compute_layer(hidden_size):
    def sru_compute_layer(u, x, bias, c0=None):
        time_size = x.size(0)
        batch_size = x.size(-2)

        k = u.size(-1) // hidden_size
        u = u.view(time_size, batch_size, hidden_size, k)

        x_ = u[..., 0]

        forget_bias, reset_bias = bias.view(2, hidden_size)

        forget = (u[..., 1] + forget_bias).sigmoid()
        reset = (u[..., 2] + reset_bias).sigmoid()

        if k == 3:
            x_prime = x
        else:
            x_prime = u[..., 3]

        h = x.new_empty(time_size, batch_size, hidden_size)

        if c0 is None:
            c_init = x.new_zeros(batch_size, hidden_size)
        else:
            c_init = c0.view(batch_size, hidden_size)

        time_seq = range(time_size)
        c_prev = c_init

        for t in time_seq:
            c_t = x_[t] * (1 - forget[t]) + c_prev * forget[t]
            g_c_t = c_t.tanh()
            h[t] = x_prime[t] * (1 - reset[t]) + g_c_t * reset[t]
            c_prev = c_t

        return h, c_t.view(batch_size, -1)

    return sru_compute_layer


class SRUCell(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size):
        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        k = 4 if input_size != hidden_size else 3
        self.k = k
        self.size_per_dir = hidden_size * k

        self.weight = nn.Parameter(torch.Tensor(input_size, self.size_per_dir))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 2))

        self.init_weight()

    def init_weight(self):
        # initialize weights such that E[w_ij]=0 and Var[w_ij]=1/d
        val_range = (3.0 / self.input_size)**0.5
        self.weight.data.uniform_(-val_range, val_range)

        # initialize bias
        self.bias.data.zero_()
        bias_val, hidden_size = 0, self.hidden_size
        self.bias.data[hidden_size:].zero_().add_(bias_val)

    def forward(self, input_, c0=None):
        """
        Args:
            input_ (seq_length, batch, input_size): Tensor containing input features.
            c0 (batch, hidden_size): Tensor containing the initial hidden state for
                each element in the batch.
        """
        assert input_.dim() == 2 or input_.dim() == 3
        input_size, hidden_size = self.input_size, self.hidden_size
        batch_size = input_.size(-2)

        if c0 is None:
            c0 = input_.new_zeros(batch_size, hidden_size)

        x = input_

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, input_size)
        u = x_2d.mm(self.weight)

        sru_compute = _sru_compute_layer(hidden_size)
        return sru_compute(u, input_, self.bias, c0)


class SRU(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers=2,
    ):
        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.out_size = hidden_size

        for i in range(num_layers):
            layer = SRUCell(
                input_size=self.input_size if i == 0 else self.out_size,
                hidden_size=self.hidden_size
            )
            self.layers.append(layer)

    def forward(self, input_, c0=None):
        """
        Args:
            input_ (seq_length, batch, input_size): Tensor containing input features.
            c0 (torch.num_layers, batch, hidden_size): Tensor containing the initial
                hidden state for each element in the batch.
        """
        assert input_.dim() == 3  # (len, batch, input_size)

        if c0 is None:
            zeros = input_.new_zeros(input_.size(1), self.hidden_size)
            c0 = [zeros for i in range(self.num_layers)]
        else:
            assert c0.dim() == 3  # (num_layers, batch, hidden_size)
            c0 = [x.squeeze(0) for x in c0.chunk(self.num_layers, 0)]

        prev_x = input_
        cs = []
        for i, rnn_layer in enumerate(self.layers):
            h, c = rnn_layer(prev_x, c0[i])
            prev_x = h
            cs.append(c)

        return prev_x, torch.stack(cs)
