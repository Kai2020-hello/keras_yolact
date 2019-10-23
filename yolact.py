from tensorflow import keras.layers as KL
from tensorflow import keras.models as KM
import keras_resnet

############################################################
#  FPN Layer
############################################################
def __create_pyramid_features(C3, C4, C5, feature_size=256):
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
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]

def _build_class_box_mask_graph(depth,num_classes,num_anchors,k):
    inputs  = KL.Input(shape=(None, None, pyramid_feature_size))
    share = keras.layers.Conv2D(filters=256,activation='relu', name='head_share_1')(inputs)
    share = keras.layers.Conv2D(filters=256,activation='relu', name='head_share_1')(share)


def _build_head(num_classes,num_anchors,k,depth):
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = _build_class_box_mask_graph(input_feature_map,num_classes,num_anchors,k)
    return KM.Model([input_feature_map], outputs, name="head_model")

############################################################
#  yolact Layer
############################################################
class yolact(object):
    def __init__(self,mode,config,model_dir):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.keras_model = self.build(mode=mode,config=config)

    def build(self,mode,config):
        """
        build yolact architecture
        """
        assert mode in ['training', 'inference']

        #输入
        img = KL.Input(shape=(None, None, 3))

        #构建backbone
        backbone = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
        C3, C4, C5 = backbone.outputs[1:]
        #构建fpn
        features = create_pyramid_features(C3, C4, C5) # 特征是 
        #构建分类，检测，mask分支
        num_classes = 1
        num_anchors = 1
        k = 1
        head_model = _build_head(num_classes,num_anchors,k,depth)
        #构建protnet 分支


        return None

