import tensorflow as tf


class ModelMaker:
    """
    Creates a model for the OpenPose project, structure is 10 layers of VGG16 followed by a few convolutions, and 6 stages
    of (PAF,PAF,PAF,PAF,kpts,kpts) also potentially includes a mask stacked with the outputs
    """

    def __init__(self, config):
        # initialize some configs which would be used in following functions
        self.IMAGE_HEIGHT = config.IMAGE_HEIGHT
        self.IMAGE_WIDTH = config.IMAGE_WIDTH
        self.PAF_NUM_FILTERS = config.PAF_NUM_FILTERS
        self.HEATMAP_NUM_FILTERS = config.HEATMAP_NUM_FILTERS
        self.BATCH_NORMALIZATION_ON = config.BATCH_NORMALIZATION_ON
        self.DROPOUT_RATE = config.DROPOUT_RATE

        self.INCLUDE_MASK = config.INCLUDE_MASK
        self.LABEL_HEIGHT = config.LABEL_HEIGHT
        self.LABEL_WIDTH = config.LABEL_WIDTH
        self.INPUT_SHAPE_ALL = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 4)
        self.INPUT_SHAPE_RGB = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3)
        self.INPUT_SHAPE_D = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1)
        self.MASK_SHAPE = (config.LABEL_HEIGHT, config.LABEL_WIDTH, 1)

        self.stage_final_nfilters = 512
        self.base_activation = tf.keras.layers.ReLU
        self.base_activation_kwargs = {'max_value': 6}

        self._get_mobilenet_layer_config_weights()

    def _get_mobilenet_layer_config_weights(self):
        # read from MobileNetV2 and save bottom 7 layers' configs 
        mobilenet_input_model = tf.keras.applications.MobileNetV2(weights='imagenet', 
                                                                  include_top=False, input_shape=self.INPUT_SHAPE_RGB)
        name_last_layer = "block_5_add"

        self.mobilenet_layers = []

        for layer in mobilenet_input_model.layers[1:]:
            layer_info = {
                    "config"   : layer.get_config()
                    , "weights": layer.get_weights()
                    , "type"   : type(layer)
                    }
            self.mobilenet_layers.append(layer_info)
            
            if layer.name == name_last_layer:
                break
        del mobilenet_input_model

    def _make_mobilenet_input_model(self, x):
        # reconstruct bottom 7 layers of MobileNetV2 from configs
        for layer_info in self.mobilenet_layers:
            copy_layer = layer_info["type"].from_config(layer_info["config"])  # the only way to make .from_config work
            
            if (layer_info['config']['name'] == 'block_1_project_BN') or (layer_info['config']['name'] =='block_3_project_BN') or (layer_info['config']['name'] =='block_5_project_BN'):
                x = copy_layer(x) # required for the proper sizing of the layer, set_weights will not work without it
                temp = x
            elif (layer_info['config']['name'] == 'block_2_add') or (layer_info['config']['name'] == 'block_4_add') or (layer_info['config']['name'] == 'block_5_add'):
                x = copy_layer([x, temp])
            
            else:
                x = copy_layer(x)
            copy_layer.set_weights(layer_info["weights"])
        return x

    def _make_mobilenetV2_conv(self, x, input_c, output_c, name, s=1, t=6):
        initializer = tf.initializers.VarianceScaling(scale=2.0)
        
        if s == 1 and input_c == output_c: temp = x
        
        # expand part
        x = tf.keras.layers.Conv2D(input_c*t, 1, use_bias=False, kernel_initializer=initializer, name=name+'_expand_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=name+'_expand_bn')(x)
        x = self.base_activation(**self.base_activation_kwargs, name=name + "_expand_relu")(x)
        
        # depthwise conv part
        x = tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False, strides=s,
                                            depthwise_initializer=initializer, name=name+'_depthwise_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=name+'_depthwise_bn')(x)
        x = self.base_activation(**self.base_activation_kwargs, name=name + "_depthwise_relu")(x)
        
        # project part
        x = tf.keras.layers.Conv2D(output_c, 1, use_bias=False, kernel_initializer=initializer, name=name+'_project_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=name+'_project_bn')(x)
        
        if s == 1 and input_c == output_c: x = tf.keras.layers.Add(name=name+'_add')([x, temp])
        
        return x
    
    def _make_mobilenetV1_conv(self, x, output_c, name):
        initializer = tf.initializers.VarianceScaling(scale=2.0)
        # depthwise conv part
        x = tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False, 
                                            depthwise_initializer=initializer, name=name+'_depthwise_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=name+'_depthwise_bn')(x)
        x = self.base_activation(**self.base_activation_kwargs, name=name + "_depthwise_relu")(x)
        
        # project part
        x = tf.keras.layers.Conv2D(output_c, 1, use_bias=False, kernel_initializer=initializer, name=name+'_project_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=name+'_project_bn')(x)
        x = self.base_activation(**self.base_activation_kwargs, name=name + "_project_relu")(x)
        
        return x
        
        
    def _make_stage0(self, x):
        # make stem stage: 7 layers of MobileNetV2 - DepthwiseConv64 - DepthwiseConv128
        
        x = self._make_mobilenetV2_conv(x, 32, 64, 'block_6')
        x = self._make_mobilenetV2_conv(x, 64, 128, 'block_7')
        
        return x

    def _make_conv_block(self, x0, conv_block_filters, name):
        # make conv block: conv-conv-conv-concat, self.BATCH_NORMALIZATION_ON controls whether to add BN layers
        
        x1 = self._make_mobilenetV1_conv(x0, conv_block_filters, name+'_conv_1')
        x2 = self._make_mobilenetV1_conv(x1, conv_block_filters, name+'_conv_2')
        x3 = self._make_mobilenetV1_conv(x2, conv_block_filters, name+'_conv_3')

        output = tf.keras.layers.concatenate([x1, x2, x3], name=name + "_output")
        return output

    def _make_stage_i(self, inputs, name, conv_block_filters, outputs, last_activation):
        # make stage: conv_block - conv_block - conv_block - conv_block - conv_block - conv_final - conv_output
        initializer = tf.initializers.VarianceScaling(scale=2.0)
        if len(inputs) > 1:
            x = tf.keras.layers.concatenate(inputs, name=name + "_input")
        else:
            x = inputs[0]
            
        if self.DROPOUT_RATE > 0: x = tf.keras.layers.Dropout(self.DROPOUT_RATE)(x)
        x = self._make_conv_block(x, conv_block_filters, name + "_block1")
        if self.DROPOUT_RATE > 0: x = tf.keras.layers.Dropout(self.DROPOUT_RATE)(x)
        x = self._make_conv_block(x, conv_block_filters, name + "_block2")
        if self.DROPOUT_RATE > 0: x = tf.keras.layers.Dropout(self.DROPOUT_RATE)(x)
        x = self._make_conv_block(x, conv_block_filters, name + "_block3")
        if self.DROPOUT_RATE > 0: x = tf.keras.layers.Dropout(self.DROPOUT_RATE)(x)
        x = self._make_conv_block(x, conv_block_filters, name + "_block4")
        if self.DROPOUT_RATE > 0: x = tf.keras.layers.Dropout(self.DROPOUT_RATE)(x)
        x = self._make_conv_block(x, conv_block_filters, name + "_block5")

        x = tf.keras.layers.Conv2D(self.stage_final_nfilters, 1, use_bias=False, 
                                   kernel_initializer=initializer, name=name+'_final_project')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=name+'_final_project_bn')(x)
        x = self.base_activation(**self.base_activation_kwargs, name=name + "_final_project_relu")(x)
        x = tf.keras.layers.Conv2D(outputs, 1, padding="same", kernel_initializer=initializer, 
                                   activation=last_activation, name=name + "_finalconv")(x)

        return x

    @staticmethod
    def rename_outputs(pre_outputs):
        new_outputs = []
        for pre_output in pre_outputs:
            new_outputs.append(
                    tf.keras.layers.Lambda(lambda x: x, name=pre_output.name.split("_")[0] + "_output")(pre_output)
                    )
        return new_outputs

    @staticmethod
    def _psd_zero_mask_to_outputs(outputs, mask_input):
        new_outputs = []
        for i, output in enumerate(outputs):
            name = output.name.split("/")[0] + "_mask"
            new_outputs.append(
                    tf.keras.layers.concatenate([output, mask_input], axis=-1, name=name)  # concat the mask to the output, at idx 0
                    )
        return new_outputs
    
    def _make_disp_stage0(self, x):
        initializer = tf.initializers.VarianceScaling(scale=2.0)
        x = tf.keras.layers.Conv2D(32, 3, use_bias=False,  strides=2, padding='same',
                                   kernel_initializer=initializer, name='disp_input_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name='disp_input_conv_bn')(x)
        x = self.base_activation(**self.base_activation_kwargs, name='disp_input_conv_relu')(x)
        x = self._make_mobilenetV2_conv(x, 32, 16, 'disp_conv_block_1', s=1, t=1)
        x = self._make_mobilenetV2_conv(x, 16, 24, 'disp_conv_block_2', s=2, t=6)
        x = self._make_mobilenetV2_conv(x, 24, 24, 'disp_conv_block_3', s=1, t=6)
        x = self._make_mobilenetV2_conv(x, 24, 32, 'disp_conv_block_4', s=2, t=6)
        x = self._make_mobilenetV2_conv(x, 32, 32, 'disp_conv_block_5', s=1, t=6)
        x = self._make_mobilenetV2_conv(x, 32, 32, 'disp_conv_block_6', s=1, t=6)
        
        return x
        

    def create_models(self):
        input_tensor = tf.keras.layers.Input(shape=self.INPUT_SHAPE_ALL)  # first layer of the model

        # mask_string="_pre_mask" if INCLUDE_MASK else ""

        # stage 00 (i know)
        stage00_output = self._make_mobilenet_input_model(input_tensor[...,:3])
        # stage 0 (2conv)
        stage0_rgb = self._make_stage0(stage00_output)
        stage0_disp = self._make_disp_stage0(input_tensor[...,-1:])
        stage0_output = tf.keras.layers.Concatenate(3, name='stage_0_concat')([stage0_rgb, stage0_disp])
        
        # PAF stages
        # stage 1
        stage1_output = self._make_stage_i([stage0_output], "s1pafs", 96, self.PAF_NUM_FILTERS, tf.keras.activations.linear)
        # stage 2
        stage2_output = self._make_stage_i([stage1_output, stage0_output], "s2pafs", 128, self.PAF_NUM_FILTERS, tf.keras.activations.linear)
        # stage 3
        stage3_output = self._make_stage_i([stage2_output, stage0_output], "s3pafs", 128, self.PAF_NUM_FILTERS, tf.keras.activations.linear)
        # stage 4
        stage4_output = self._make_stage_i([stage3_output, stage0_output], "s4pafs", 128, self.PAF_NUM_FILTERS, tf.keras.activations.linear)
        # keypoint heatmap stages
        # stage5
        stage5_output = self._make_stage_i([stage4_output, stage0_output], "s5kpts", 96, self.HEATMAP_NUM_FILTERS, tf.keras.activations.tanh)
        # stage6
        stage6_output = self._make_stage_i([stage5_output, stage4_output, stage0_output], "s6kpts", 128, self.HEATMAP_NUM_FILTERS, tf.keras.activations.tanh)

        training_inputs = input_tensor
        training_outputs = [stage1_output, stage2_output, stage3_output, stage4_output, stage5_output, stage6_output]

        if self.INCLUDE_MASK:  # this is used to pass the mask directly to the loss function through the model
            mask_input = tf.keras.layers.Input(shape=self.MASK_SHAPE)
            training_outputs = self._psd_zero_mask_to_outputs(training_outputs, mask_input)
            training_inputs = (input_tensor, mask_input)

        training_outputs = self.rename_outputs(training_outputs)

        train_model = tf.keras.Model(inputs=training_inputs, outputs=training_outputs)

        test_outputs = [stage4_output, stage6_output]
        test_model = tf.keras.Model(inputs=input_tensor, outputs=test_outputs)

        return train_model, test_model


class ModelDatasetComponent:
    def __init__(self, config):
        self.INCLUDE_MASK = config.INCLUDE_MASK

    @tf.function
    def place_training_labels(self, elem):
        """Distributes labels into the correct configuration for the model, ie 4 PAF stage, 2 kpt stages
        must match the model"""
        paf_tr = elem['pafs']
        kpt_tr = elem['kpts']
        image = elem['image']

        if self.INCLUDE_MASK:
            inputs = (image, elem['mask'])
        else:
            inputs = image
        return inputs, (paf_tr, paf_tr, paf_tr, paf_tr, kpt_tr, kpt_tr)  # this should match the model outputs, and is different for each model
