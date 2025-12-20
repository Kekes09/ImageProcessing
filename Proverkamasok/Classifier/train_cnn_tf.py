# Classifier/train_cnn_tf.py
import os, json
import tensorflow as tf

TRAIN_DIR = "data/Train"
VAL_DIR   = "data/Validation"
TEST_DIR  = "data/Test"

IMG_SIZE = (224, 224)
BATCH = 32

MODEL_OUT = "models/cnn_mask.keras"
LABELS_OUT = "models/labels.json"

def make_ds(path, shuffle=False):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=shuffle,
    )

def main():
    os.makedirs("models", exist_ok=True)

    train_ds = make_ds(TRAIN_DIR, shuffle=True)
    val_ds   = make_ds(VAL_DIR, shuffle=False)
    test_ds  = make_ds(TEST_DIR, shuffle=False)

    class_names = train_ds.class_names  # порядок важен!
    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names}, f, ensure_ascii=False, indent=2)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = preprocess(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                 tf.keras.metrics.Precision(name="prec"),
                 tf.keras.metrics.Recall(name="rec")]
    )

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_acc", save_best_only=True),
    ]

    print("Stage 1: train head")
    model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=cb)

    print("Stage 2: fine-tune")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                 tf.keras.metrics.Precision(name="prec"),
                 tf.keras.metrics.Recall(name="rec")]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=6, callbacks=cb)

    print("Evaluate on test:")
    best = tf.keras.models.load_model(MODEL_OUT)
    res = best.evaluate(test_ds, verbose=1)
    print(dict(zip(best.metrics_names, res)))
    print("Saved:", MODEL_OUT)

if __name__ == "__main__":
    main()
