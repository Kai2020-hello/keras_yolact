from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from config import config
import tensorflow as tf

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
    x = BatchNorm(name=bn_name_base + '2a')(x, training=True)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=True)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=True)

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
    share = BatchNorm(name='head_share_bn' + '1')(share, training=True)
    

    #分类分支
    outputs_classification = KL.Conv2D(filters=256,kernel_size=3, strides=1, padding='same',activation='relu', name='pyramid_classification_1')(share)
    outputs_classification = BatchNorm(name='pyramid_classification_share_' + '1')(outputs_classification, training=True)
    outputs_classification = KL.Conv2D(num_anchors * num_classes,kernel_size=3, strides=1, padding='same', name='pyramid_classification_2')(outputs_classification) #todo 确定参数
    outputs_classification = KL.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs_classification)
    outputs_classification = KL.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs_classification)

    #回归分支
    outputs_regression = KL.Conv2D(filters=256,kernel_size=3, strides=1, padding='same',activation='relu', name='outputs_regression_1')(share)
    outputs_regression = BatchNorm(name='outputs_regression_share_' + '1')(outputs_regression, training=True)
    outputs_regression = KL.Conv2D(num_anchors * 4,kernel_size=3, strides=1, padding='same', name='pyramid_regression')(outputs_regression) #todo 确定参数
    outputs_regression = KL.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs_regression)

    #mask coefficients 分支
    outputs_mask = KL.Conv2D(filters=256,kernel_size=3, strides=1, padding='same',activation='relu', name='outputs_mask_1')(share)
    outputs_mask = BatchNorm(name='outputs_mask_share_' + '1')(outputs_mask, training=True)
    outputs_mask = KL.Conv2D(num_anchors * k,kernel_size=3, strides=1, padding='same', name='pyramid_mask_coefficients',activation='tanh')(outputs_mask) #todo 确定参数
    outputs_mask = KL.Reshape((-1, k), name='pyramid_mask_coefficients_reshape')(outputs_mask)

    #result = KL.Concatenate(axis=1)([outputs_classification,outputs_regression,outputs_mask])
    return KM.Model(inputs,[outputs_classification,outputs_regression,outputs_mask])


############################################################
#  protnet Layer
############################################################

def build_protnet(depth,k):
    inputs  = KL.Input(shape=(None, None, depth))
    outputs = KL.Conv2D(256, kernel_size=3, strides=1, padding='same', name='rotnet_cov_1')(inputs)
    outputs = BatchNorm(name='protnet_bn' + '1a')(outputs, training=True)
    outputs = KL.Conv2D(256, kernel_size=1, strides=1, padding='same', name='rotnet_cov_2')(outputs)
    outputs = BatchNorm(name='protnet_bn' + '2a')(outputs, training=True)

    outputs = KL.UpSampling2D(size=(2, 2),name='rotnet_upsampling_1')(outputs)
    outputs = KL.Conv2D(256, kernel_size=3, strides=1, padding='same', name='rotnet_cov_3')(outputs)
    outputs = BatchNorm(name='protnet_bn' + '1b')(outputs, training=True)
    outputs = KL.Conv2D(256, kernel_size=1, strides=1, padding='same', name='rotnet_cov_4')(outputs)
    outputs = BatchNorm(name='protnet_bn' + '2b')(outputs, training=True)

    outputs = KL.UpSampling2D(size=(2, 2),name='rotnet_upsampling_2')(outputs)
    outputs = KL.Conv2D(256, kernel_size=3, strides=1, padding='same', name='rotnet_cov_5')(outputs)
    outputs = BatchNorm(name='protnet_bn' + '1c')(outputs, training=True)
    outputs = KL.Conv2D(256, kernel_size=1, strides=1, padding='same', name='rotnet_cov_6')(outputs)
    outputs = BatchNorm(name='protnet_bn' + '2c')(outputs, training=True)

    outputs = KL.UpSampling2D(size=(2, 2),name='rotnet_upsampling_2')(outputs)
    outputs = KL.Conv2D(k, kernel_size=1, strides=1, padding='same', name='rotnet_cov_5',activation='relu')(outputs)

    return KM.Model(inputs,outputs)


############################################################
#  nms Layer
############################################################
def filter_detections(
    boxes,
    classification,
    other                 = [],
    class_specific_filter = True,
    nms                   = True,
    score_threshold       = 0.05,
    max_detections        = 300,
    nms_threshold         = 0.5
):
    """ Filter detections using the boxes and classification values.

    Args
        boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other                 : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        nms                   : Flag to enable/disable non maximum suppression.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    def _filter_detections(scores, labels):
        # threshold based on score
        indices = tf.where(tfgreater(scores, score_threshold))

        if nms:
            filtered_boxes  = tf.gather_nd(boxes, indices)
            filtered_scores = tf.gather(scores, indices)[:, 0]

            # perform NMS
            nms_indices = tf.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)

            # filter indices based on NMS
            indices = tf.gather(indices, nms_indices)

        # add indices to list of all indices
        labels = tf.gather_nd(labels, indices)
        indices = tf.stack([indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((tf.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        indices = tf.concatenate(all_indices, axis=0)
    else:
        scores  = tf.max(classification, axis    = 1)
        labels  = tf.argmax(classification, axis = 1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores              = tf.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = tf.top_k(scores, k=tf.minimum(max_detections, tf.shape(scores)[0]))

    # filter input using the final set of indices
    indices             = tf.gather(indices[:, 0], top_indices)
    boxes               = tf.gather(boxes, indices)
    labels              = tf.gather(labels, top_indices)
    other_              = [tf.gather(o, indices) for o in other]

    # zero pad the outputs
    pad_size = tf.maximum(0, max_detections - tf.shape(scores)[0])
    boxes    = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = tf.cast(labels, 'int32')
    other_   = [tf.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(tf.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels] + other_


class FilterDetections(KL.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
        self,
        nms                   = True,
        class_specific_filter = True,
        nms_threshold         = 0.5,
        score_threshold       = 0.05,
        max_detections        = 300,
        parallel_iterations   = 32,
        **kwargs
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms                   : Flag to enable/disable NMS.
            class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold       : Threshold used to prefilter the boxes with.
            max_detections        : Maximum number of detections to keep.
            parallel_iterations   : Number of batch items to process in parallel.
        """
        self.nms                   = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold         = nms_threshold
        self.score_threshold       = score_threshold
        self.max_detections        = max_detections
        self.parallel_iterations   = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes          = inputs[0]
        classification = inputs[1]
        other          = inputs[2:]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes          = args[0]
            classification = args[1]
            other          = args[2]

            return filter_detections(
                boxes,
                classification,
                other,
                nms                   = self.nms,
                class_specific_filter = self.class_specific_filter,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
                nms_threshold         = self.nms_threshold,
            )

        # call filter_detections on each batch
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification, other],
            dtype=[tf.floatx(), tf.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ] + [
            tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))
        ]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms'                   : self.nms,
            'class_specific_filter' : self.class_specific_filter,
            'nms_threshold'         : self.nms_threshold,
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : self.parallel_iterations,
        })

        return config

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

        #对边框和mask做nms
        boxes, scores, labels , mask_coefficients = FilterDetections(nms= True,class_specific_filter = True,name = 'filtered_detections')([boxes_concat, classes_concat] + masks_concat)

        #构建protnet部分
        protnet = build_protnet(depth,k)
        mask_protype = protnet(features[0])

        # mask Assembly
        mask = tf.sigmoid(mask_protype * tf.transpose(mask_coefficients,[0,2,1]))

        return KM.Model(img,[boxes, scores, labels,mask])


if __name__ == "__main__":
    config = config()
    yolact = yolact('training',config,"test")

