from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from config import config

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

############################################################
#  Resnet Graph
############################################################
# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    return x


def resnet_graph(input_image, architecture, stage5=True, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C2, C3, C4, C5]


############################################################
#  FPN Layer
############################################################
def _create_pyramid_features(C3, C4, C5, feature_size=256):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5           = KL.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = KL.UpSampling2D(size=(2, 2),name='P5_upsampled')(P5)
    P5           = KL.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = KL.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = KL.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = KL.UpSampling2D(size=(2, 2),name='P4_upsampled')(P4)
    P4           = KL.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = KL.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = KL.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = KL.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = KL.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = KL.Activation('relu', name='C6_relu')(P6)
    P7 = KL.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]

def _build_class_box_mask_graph(depth,num_classes,num_anchors,k):
    inputs  = KL.Input(shape=(None, None, depth)) # todo
    share = KL.Conv2D(filters=256,kernel_size=3, strides=1, padding='same',activation='relu', name='head_share_1')(inputs)
    share = KL.Conv2D(filters=256,kernel_size=3, strides=1, padding='same',activation='relu', name='head_share_2')(share)

    #分类分支
    outputs_classification = KL.Conv2D(num_anchors * num_classes,kernel_size=3, strides=1, padding='same', name='pyramid_classification_')(share) #todo 确定参数
    outputs_classification = KL.Reshape((-1, num_classes), name='pyramid_classification__reshape')(outputs_classification)

    #回归分支
    outputs_regression = KL.Conv2D(num_anchors * 4,kernel_size=3, strides=1, padding='same', name='pyramid_regression')(share) #todo 确定参数
    outputs_regression = KL.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs_regression)

    #mask coefficients 分支
    outputs_mask = KL.Conv2D(num_anchors * k,kernel_size=3, strides=1, padding='same', name='pyramid_mask_coefficients',activation='tanh')(share) #todo 确定参数
    outputs_mask = KL.Reshape((-1, k), name='pyramid_mask_coefficients_reshape')(outputs_mask)

    #result = KL.Concatenate(axis=1)([outputs_classification,outputs_regression,outputs_mask])
    return KM.Model(inputs,[outputs_classification,outputs_regression,outputs_mask])


############################################################
#  protnet Layer
############################################################

def build_protnet(depth,k):
    inputs  = KL.Input(shape=(None, None, depth))
    outputs = KL.Conv2D(256, kernel_size=3, strides=1, padding='same', name='rotnet_1')(inputs)
    outputs = KL.Conv2D(256, kernel_size=1, strides=1, padding='same', name='rotnet_2')(inputs)
    outputs = KL.UpSampling2D(size=(2, 2),name='rotnet_upsampling_1')(outputs)
    outputs = KL.Conv2D(k, kernel_size=1, strides=1, padding='same', name='rotnet_3')(inputs)
    return KM.Model(inputs,outputs)


############################################################
#  yolact model
############################################################
class yolact(object):
    def __init__(self,mode,config,model_dir):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.keras_model = self.build(mode=mode,config=config)
        self.keras_model.summary()

    def build(self,mode,config):
        """
        build yolact architecture
        """
        assert mode in ['training', 'inference']

        #输入
        img = KL.Input(shape=(None, None, 3))

        #构建backbone
        _, C3, C4, C5 = resnet_graph(img, "resnet50")
        #构建fpn
        features = _create_pyramid_features(C3, C4, C5) # 特征是 
        #构建分类，检测，mask分支
        num_classes = 10
        num_anchors = 10
        k = 10
        depth = 256
        head_model = _build_class_box_mask_graph(depth,num_classes,num_anchors,k)
        classes_concat = []
        boxes_concat = []
        masks_concat = []
        for feature in features:
            classes,boxes,masks = head_model(feature)
            classes_concat.append(classes)
            boxes_concat.append(boxes)
            masks_concat.append(masks)

        classes_concat = KL.Concatenate(axis=1)(classes_concat)
        boxes_concat = KL.Concatenate(axis=1)(boxes_concat)
        masks_concat = KL.Concatenate(axis=1)(masks_concat)
        #构建protnet部分
        protnet = build_protnet(depth,k)
        mask_protype = protnet(features[0])


        return KM.Model(img,[classes_concat,boxes_concat,masks_concat,mask_protype])

if __name__ == "__main__":
    config = config()
    yolact = yolact('training',config,"test")

