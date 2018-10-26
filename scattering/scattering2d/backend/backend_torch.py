import torch
from torch.legacy.nn import SpatialReflectionPadding as pad_function

NAME = 'torch'


def iscomplex(input):
    return input.size(-1) == 2


# This function copies and view the real to complex
def pad(input, pre_pad):
    if(pre_pad):
        output = input.new(input.size(0), input.size(1), input.size(2), input.size(3), 2).fill_(0)
        output.narrow(output.ndimension()-1, 0, 1).copy_(input)
    else:
        out_ = self.padding_module.updateOutput(input)
        output = input.new(*(out_.size() + (2,))).fill_(0)
        output.select(4, 0).copy_(out_)
    return output

def unpad(self, in_):
    return in_[..., 1:-1, 1:-1]

class Periodize(object):
    """This class builds a wrapper to the periodiziation kernels and cache them.
        """

    def __call__(self, input, k):
        out = input.new(input.size(0), input.size(1), input.size(2) // k, input.size(3) // k, 2)


        y = input.view(input.size(0), input.size(1),
                       input.size(2)//out.size(2), out.size(2),
                       input.size(3)//out.size(3), out.size(3),
                       2)

        out = y.mean(4, keepdim=False).mean(2, keepdim=False)
        return out


class Modulus(object):
    """This class builds a wrapper to the moduli kernels and cache them.
        """

    def __call__(self, input):

        norm = input.norm(p=2, dim=-1, keepdim=True)
        return torch.cat([norm, torch.zeros_like(norm)], -1)



class Fft(object):
    """This class builds a wrapper to the FFTs kernels and cache them.

    As a try, the library will purely work with complex data. The FFTS are UNORMALIZED.
        """
    def __call__(self, input, direction='C2C', inverse=False):
        if direction == 'C2R':
            inverse = True

        if not iscomplex(input):
            raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensors must be contiguous!'))

        if direction == 'C2R':
            output = torch.irfft(input, 2, normalized=False, onesided=False)*input.size(-2)*input.size(-3)
        elif direction == 'C2C':
            if inverse:
                output = torch.ifft(input, 2, normalized=False)*input.size(-2)*input.size(-3)
            else:
                output = torch.fft(input, 2, normalized=False)

        return output




def cdgmm(A, B, inplace=False):
    """This function uses the C-wrapper to use cuBLAS.
        """
    A, B = A.contiguous(), B.contiguous()
    if A.size()[-3:] != B.size():
        raise RuntimeError('The filters are not compatible for multiplication!')

    if not iscomplex(A) or not iscomplex(B):
        raise TypeError('The input, filter and output should be complex')

    if B.ndimension() != 3:
        raise RuntimeError('The filters must be simply a complex array!')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type!')


    C = A.new(A.size())

    A_r = A[..., 0].contiguous().view(-1, A.size(-2)*A.size(-3))
    A_i = A[..., 1].contiguous().view(-1, A.size(-2)*A.size(-3))

    B_r = B[...,0].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_i)
    B_i = B[..., 1].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_r)

    C[..., 0].view(-1, C.size(-2)*C.size(-3))[:] = A_r * B_r - A_i * B_i
    C[..., 1].view(-1, C.size(-2)*C.size(-3))[:] = A_r * B_i + A_i * B_r

    return C if not inplace else A.copy_(C)



