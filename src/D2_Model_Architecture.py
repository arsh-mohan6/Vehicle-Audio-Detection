
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers


# ==========================================
# 1. MODEL SETTINGS
# ==========================================

NUM_CLASSES = 5

MFCC_SHAPE = (40, 216, 1)
MEL_SHAPE = (128, 216, 1)

L2_VALUE = 0.001


# ==========================================
# 2. CNN BRANCH
# ==========================================

def create_cnn_branch(input_layer, branch_name):

    x = input_layer

    filters = [32, 64, 128]

    for i, filter_count in enumerate(filters):

        x = layers.Conv2D(
            filters=filter_count,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(L2_VALUE),
            name=f"{branch_name}_conv_{i+1}"
        )(x)

        x = layers.BatchNormalization(
            name=f"{branch_name}_bn_{i+1}"
        )(x)

        x = layers.Activation(
            "relu",
            name=f"{branch_name}_relu_{i+1}"
        )(x)

        x = layers.MaxPooling2D(
            pool_size=(2, 2),
            name=f"{branch_name}_pool_{i+1}"
        )(x)

        x = layers.Dropout(
            0.20,
            name=f"{branch_name}_dropout_{i+1}"
        )(x)

    # Convert feature maps into a feature vector
    x = layers.GlobalAveragePooling2D(
        name=f"{branch_name}_global_pool"
    )(x)

    # Dense feature representation
    x = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(L2_VALUE),
        name=f"{branch_name}_dense"
    )(x)

    x = layers.BatchNormalization(
        name=f"{branch_name}_dense_bn"
    )(x)

    x = layers.Activation(
        "relu",
        name=f"{branch_name}_dense_relu"
    )(x)

    x = layers.Dropout(
        0.30,
        name=f"{branch_name}_dense_dropout"
    )(x)

    return x


# ==========================================
# 3. BUILD DUAL-BRANCH CNN MODEL
# ==========================================

def build_model():

    # MFCC input
    mfcc_input = layers.Input(
        shape=MFCC_SHAPE,
        name="mfcc_input"
    )

    # Mel input
    mel_input = layers.Input(
        shape=MEL_SHAPE,
        name="mel_input"
    )

    # Create MFCC branch
    mfcc_features = create_cnn_branch(
        mfcc_input,
        "mfcc"
    )

    # Create Mel branch
    mel_features = create_cnn_branch(
        mel_input,
        "mel"
    )

    # ======================================
    # 4. FEATURE FUSION
    # ======================================

    combined = layers.Concatenate(
        name="feature_fusion"
    )([
        mfcc_features,
        mel_features
    ])

    # ======================================
    # 5. CLASSIFICATION LAYERS
    # ======================================

    x = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(L2_VALUE),
        name="fusion_dense_1"
    )(combined)

    x = layers.BatchNormalization(
        name="fusion_bn_1"
    )(x)

    x = layers.Activation(
        "relu",
        name="fusion_relu_1"
    )(x)

    x = layers.Dropout(
        0.50,
        name="fusion_dropout_1"
    )(x)

    x = layers.Dense(
        64,
        kernel_regularizer=regularizers.l2(L2_VALUE),
        name="fusion_dense_2"
    )(x)

    x = layers.BatchNormalization(
        name="fusion_bn_2"
    )(x)

    x = layers.Activation(
        "relu",
        name="fusion_relu_2"
    )(x)

    x = layers.Dropout(
        0.30,
        name="fusion_dropout_2"
    )(x)

    # ======================================
    # 6. OUTPUT LAYER
    # ======================================

    output = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="vehicle_prediction"
    )(x)

    # ======================================
    # 7. CREATE FINAL MODEL
    # ======================================

    model = models.Model(
        inputs=[mfcc_input, mel_input],
        outputs=output,
        name="Vehicle_Audio_Dual_Branch_CNN"
    )

    return model


# ==========================================
# 8. TEST MODEL ARCHITECTURE
# ==========================================

def main():

    print("\n===================================")
    print(" VEHICLE AUDIO - D2 ARCHITECTURE")
    print("===================================")

    # Build model
    model = build_model()

    # Display model summary
    print("\nMODEL SUMMARY:\n")

    model.summary()

    # Check model input shapes
    print("\nINPUT SHAPES:")

    for model_input in model.inputs:
        print(model_input.name, model_input.shape)

    # Check output shape
    print("\nOUTPUT SHAPE:")
    print(model.output_shape)

    # Count parameters
    print("\nTOTAL PARAMETERS:")
    print(model.count_params())

    # Test with dummy input
    print("\nTESTING MODEL WITH DUMMY DATA...")

    dummy_mfcc = tf.random.normal(
        (1, 40, 216, 1)
    )

    dummy_mel = tf.random.normal(
        (1, 128, 216, 1)
    )

    predictions = model(
        [dummy_mfcc, dummy_mel],
        training=False
    )

    print("Prediction shape:", predictions.shape)

    print(
        "Probability sum:",
        float(tf.reduce_sum(predictions[0]))
    )

    print("\n===================================")
    print(" D2 ARCHITECTURE COMPLETED!")
    print("===================================")


if __name__ == "__main__":
    main()
