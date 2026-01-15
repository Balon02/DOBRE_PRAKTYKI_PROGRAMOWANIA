import math
import keras
from keras import layers, ops, backend, activations

from typing import Callable

DEFAULT_CONV_ACT = 'silu' # silulululululu :^)

# !!!!ALL THE LAYERS ASSUME CHANNELS IN THE LAST DIMENTION INSTEAD OF 2ND WHICH WAS THE CASE IN ORIGINAL IMPLEMENTATION!!!!
# THIS IS ESPECIALLY IMPORTANT FOR THE ATTENTION MODULE, AS THE COLLAPSE OF WIDTH x HEIGHT AND RESHAPE OF CHANNEL DIM CHANGE THE RESHAPING LOGIC FROM ORIGINAL ONE TO RETAIN FLASH ATTENTION EXECUTION SPEED
# ALL OTHER LAYERS SHOULD HANDLE THIS CHANGE SEAMLESSLY AS CHANNELS LAST IS THE DEFAULT KERAS ASSUMPTION

# PORTED FROM https://github.com/ultralytics/ultralytics/blob/5c226935aa6d7651d1861968f713dfba027a5089/ultralytics/nn/modules/conv.py

class YoloConv2D(layers.Layer):
    """Yolo-style wrapper for 2D convolution, with batch norm before activation defaulting to SILU/SWISH."""
    def __init__(self, filters:int, kernel_size:int=1, strides:int=1, padding:str='same', groups:int=1, dilation_rate:int=1, act:str|bool=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.dilation_rate = dilation_rate
        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups, dilation_rate=dilation_rate, use_bias=False, activation=None)
        self.norm = layers.BatchNormalization(epsilon=backend.epsilon())
        self.act = activations.get(DEFAULT_CONV_ACT) if act is True else activations.get(act) if isinstance(act, str) else keras.layers.Identity()

    def call(self, x, training=None):
        if training is None: training = False
        x = self.conv(x)
        x = self.norm(x, training=training)
        return self.act(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size, "strides": self.strides, "padding": self.padding, "groups": self.groups, "dilation_rate": self.dilation_rate, "act": self.act})
        return config
    
class YoloDWConv2D(layers.Layer):
    """Yolo-style wrapper for 2D convolution, with batch norm before activation defaulting to SILU/SWISH."""
    def __init__(self, filters:int, kernel_size:int=1, strides:int=1, padding:str='same', dilation_rate:int=1, act:str|bool=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.act = act

    def build(self, shape):
        self.conv = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, groups=math.gcd(shape[-1], self.filters), dilation_rate=self.dilation_rate, use_bias=False, activation=None)
        self.norm = layers.BatchNormalization(epsilon=backend.epsilon())
        self.act = activations.get(DEFAULT_CONV_ACT) if self.act is True else activations.get(self.act) if isinstance(self.act, str) else keras.layers.Identity()
        super().build(shape)

    def call(self, x, training=None):
        if training is None: training = False
        x = self.conv(x)
        x = self.norm(x, training=training)
        return self.act(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size, "strides": self.strides, "padding": self.padding, "dilation_rate": self.dilation_rate, "act": self.act})
        return config

# PORTED FROM https://github.com/ultralytics/ultralytics/blob/5c226935aa6d7651d1861968f713dfba027a5089/ultralytics/nn/modules/block.py

class Bottleneck(layers.Layer):
    """Base bottleneck block with no residual connection, usable if input and output shapes mismatch."""
    def __init__(self, dim:int, groups:int=1, kernel_sizes:tuple[int,int]=(3,3), expansion_ratio:float=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.groups = groups
        self.kernel_sizes = kernel_sizes
        self.expansion_ratio = expansion_ratio
        hidden_channels = int(dim * expansion_ratio)
        self.conv1 = YoloConv2D(filters=hidden_channels, kernel_size=kernel_sizes[0])
        self.conv2 = YoloConv2D(filters=dim, kernel_size=kernel_sizes[1], groups=groups)

    def call(self, x, training=None):
        if training is None: training = False
        return self.conv2(self.conv1(x, training=training), training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "groups": self.groups, "kernel_sizes": self.kernel_sizes, "expansion_ratio": self.expansion_ratio})
        return config


class ResidualBottleneck(Bottleneck):
    """Case of the bottleneck block with residual connection, usable only when output shape matches."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, training=None):
        out = super().call(x, training=training)
        return x + out
    
    
class C3k(layers.Layer):
    """CSP Bottleneck with 3 convolutions."""
    def __init__(self, dim:int, kernel_sizes:int|tuple[int, int]=3, num_bottlenecks:int = 1, shortcut:bool=False, groups:int=1, expansion_ratio:float=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.kernel_sizes = (kernel_sizes, kernel_sizes) if isinstance(kernel_sizes, int) else kernel_sizes
        self.num_bottlenecks = num_bottlenecks
        self.shortcut = shortcut
        self.groups = groups
        self.expansion_ratio = expansion_ratio
        hidden_channels = int(dim * expansion_ratio)
        self.conv1 = YoloConv2D(filters=hidden_channels, kernel_size=1, strides=1)
        self.conv2 = YoloConv2D(filters=hidden_channels, kernel_size=1, strides=1)
        self.conv3 = YoloConv2D(filters=dim, kernel_size=1, strides=1)
        bottleneck = ResidualBottleneck if shortcut else Bottleneck
        self.modules = keras.Sequential([bottleneck(dim=hidden_channels, groups=groups, kernel_sizes=self.kernel_sizes, expansion_ratio=1.0) for _ in range(num_bottlenecks)])

    def call(self, x, training=None):
        if training is None: training = False
        x1 = self.modules(self.conv1(x, training=training), training=training)
        x2 = self.conv2(x, training=training)
        return self.conv3(ops.concatenate((x1, x2), axis=-1), training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "kernel_sizes": self.kernel_sizes, "num_bottlenecks": self.num_bottlenecks, "shortcut": self.shortcut, "groups": self.groups, "expansion_ratio": self.expansion_ratio})
        return config


class C3k2(layers.Layer):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, dim:int, num_bottlenecks:int=1, shortcut:bool=False, use_c3k:bool=False, groups:int=1, expansion_ratio:float=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_bottlenecks = num_bottlenecks
        self.shortcut = shortcut
        self.use_c3k = use_c3k
        self.groups = groups
        self.expansion_ratio = expansion_ratio
        hidden_channels = int(dim * expansion_ratio)
        conv1_filters = 2 * hidden_channels
        self.conv1 = YoloConv2D(filters=conv1_filters, kernel_size=1, strides=1)
        self.conv2 = YoloConv2D(filters=dim, kernel_size=1, strides=1)
        if use_c3k:
            self.modules = [C3k(dim=hidden_channels, num_bottlenecks=2, groups=groups, shortcut=shortcut) for _ in range(num_bottlenecks)]
        else:
            bottleneck = ResidualBottleneck if shortcut else Bottleneck
            self.modules = [bottleneck(dim=hidden_channels, groups=groups) for _ in range(num_bottlenecks)]
    
    def call(self, x, training=None):
        if training is None: training = False
        x = self.conv1(x, training=training)
        x1, x2 = ops.split(x, 2, axis=-1)
        xs = [x1, x2]
        for module in self.modules: xs.append(module(xs[-1], training=training))
        return self.conv2(ops.concatenate(xs, axis=-1), training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "num_bottlenecks": self.num_bottlenecks, "shortcut": self.shortcut, "groups": self.groups, "expansion_ratio": self.expansion_ratio, "use_c3k": self.use_c3k})
        return config
        

class SPPF(layers.Layer):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
    def __init__(self, dim:int, kernel_size:int=5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.kernel_size = kernel_size

    def build(self, input_shape):
        previous_dim = input_shape[-1]
        hidden_channels = previous_dim // 2 # TODO: CHECK AGAIN WITH THE ORIGINAL IMPLEMENTATION
        self.conv1 = YoloConv2D(filters=hidden_channels, kernel_size=1, strides=1)
        self.conv2 = YoloConv2D(filters=self.dim, kernel_size=1, strides=1)
        self.pool = layers.MaxPool2D(pool_size=self.kernel_size, strides=1, padding='same')
        super().build(input_shape)

    def call(self, x, training=None):
        if training is None: training = False
        xs = [self.conv1(x, training=training)]
        xs.extend(self.pool(xs[-1]) for _ in range(3))
        return self.conv2(ops.concatenate(xs, axis=-1), training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "kernel_size": self.kernel_size})
        return config


class YoloAttention2D(layers.Layer):
    """Mathematical equivalent of ultralytics self-attention (Attention class) module rewritten to keras in channels_last format."""
    def __init__(self, dim:int, num_heads:int, attention_ratio:float=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dim=dim
        self.num_heads = num_heads
        self.attention_ratio = attention_ratio
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attention_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = YoloConv2D(filters=h, act=False)
        self.proj = YoloConv2D(filters=dim, act=False)
        self.pos_enc = YoloConv2D(filters=dim, kernel_size=3, groups=dim, act=False)

    def call(self, x: ops.array, training=None):
        if training is None: training = False
        shape = ops.shape(x)
        B = shape[0]
        H = shape[1]
        W = shape[2]
        C = shape[3]
        N = H * W
        per_head = 2 * self.key_dim + self.head_dim

        qkv = self.qkv(x, training=training)
        qkv = ops.reshape(qkv, (B, N, self.num_heads, per_head))
        qkv = ops.transpose(qkv, (0, 2, 3, 1))

        split_indices = [self.key_dim, 2 * self.key_dim]
        q, k, v = ops.split(qkv, split_indices, axis=2)

        attn = ops.matmul(ops.transpose(q, (0, 1, 3, 2)), k) * self.scale
        attn = ops.softmax(attn, axis=-1)

        context = ops.matmul(v, ops.transpose(attn, (0, 1, 3, 2)))
        context = ops.transpose(context, (0, 3, 1, 2))
        context = ops.reshape(context, (B, H, W, C))

        v_spatial = ops.transpose(v, (0, 3, 1, 2))
        v_spatial = ops.reshape(v_spatial, (B, H, W, C))
        context = context + self.pos_enc(v_spatial, training=training)
        return self.proj(context, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "num_heads": self.num_heads, "attention_ratio": self.attention_ratio})
        return config

    
class PSABlock(layers.Layer):
    def __init__(self, dim:int, num_heads:int, attention_ratio:float=0.5, shortcut:bool=True, **kwargs):
        super().__init__(**kwargs)
        self.dim=dim
        self.num_heads = num_heads
        self.attention_ratio = attention_ratio
        self.shortcut = shortcut
        self.attn = YoloAttention2D(dim=dim, num_heads=num_heads, attention_ratio=attention_ratio)
        self.ffn = keras.Sequential([YoloConv2D(filters=dim*2, kernel_size=1), YoloConv2D(filters=dim, kernel_size=1, act=False)])

    def call(self, x, training=None):
        if training is None: training = False
        x = x + self.attn(x, training=training) if self.shortcut else self.attn(x, training=training)
        x = x + self.ffn(x, training=training) if self.shortcut else self.ffn(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "num_heads": self.num_heads, "attention_ratio": self.attention_ratio, "shortcut": self.shortcut})
        return config
    

class C2PSA(layers.Layer):
    def __init__(self, dim:int, num_psa_blocks=1, expansion_ratio:float=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_psa_blocks = num_psa_blocks
        self.expansion_ratio = expansion_ratio
        self.channels = int(dim * expansion_ratio)
        self.conv1 = YoloConv2D(filters=2 * self.channels, kernel_size=1, strides=1)
        self.conv2 = YoloConv2D(filters=dim, kernel_size=1, strides=1)
        self.blocks = keras.Sequential(
            [PSABlock(self.channels, num_heads=max(1, self.channels // 64), attention_ratio=0.5) for _ in range(num_psa_blocks)]
        )

    def call(self, x, training=None):
        if training is None: training = False
        x = self.conv1(x, training=training)
        a, b = ops.split(x, 2, axis=-1)
        b = self.blocks(b, training=training)
        x = ops.concatenate([a, b], axis=-1)
        return self.conv2(x, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "num_psa_blocks": self.num_psa_blocks, "expansion_ratio": self.expansion_ratio})
        return config
    
class DFL(layers.Layer):
    """Distribution Focal Loss (DFL)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, shape):
        channels = shape[-1]
        def dfl_initializer(shape, dtype=None): return ops.reshape(ops.arange(channels, dtype=dtype), shape)
        self.conv = layers.Conv2D(filters=1, kernel_size=1, use_bias=False, trainable=False, kernel_initializer=dfl_initializer)
        super().build(shape)

    # CODEX-WRITTEN
    def call(self, x, training=None):
        if training is None: training = False
        x = ops.softmax(x, axis=-1)    # (B,N,4,reg_max)
        x = self.conv(x)               # (B,N,4,1)
        return ops.squeeze(x, axis=-1) # (B,N,4)

