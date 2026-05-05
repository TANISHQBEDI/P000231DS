import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 16
seed = 42

PREPROCESSOR_URL = "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3"
ENCODER_URL = "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-4-h-256-a-4/2"
MAX_EPOCHS = 15


def build_datasets_from_frames(train_df, val_df, batch_size_override=None):
	batch = batch_size_override or batch_size
	train_texts = train_df["text"].tolist()
	val_texts = val_df["text"].tolist()
	train_labels = train_df["label"].tolist()
	val_labels = val_df["label"].tolist()

	label_lookup = tf.keras.layers.StringLookup()
	label_lookup.adapt(train_labels)

	train_label_ids = label_lookup(train_labels)
	val_label_ids = label_lookup(val_labels)

	train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_label_ids))
	train_ds = (
		train_ds.shuffle(buffer_size=len(train_texts), seed=seed)
		.batch(batch)
		.prefetch(AUTOTUNE)
	)

	val_ds = tf.data.Dataset.from_tensor_slices((val_texts, val_label_ids))
	val_ds = val_ds.batch(batch).prefetch(AUTOTUNE)

	num_labels = int(label_lookup.vocabulary_size())
	return train_ds, val_ds, num_labels, label_lookup


def build_model(num_labels: int) -> tf.keras.Model:
	text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
	preprocessor = hub.KerasLayer(PREPROCESSOR_URL, name="preprocessor")
	encoder_inputs = preprocessor(text_input)

	encoder = hub.KerasLayer(ENCODER_URL, trainable=True, name="encoder")
	outputs = encoder(encoder_inputs)
	pooled_output = outputs["pooled_output"]

	x = tf.keras.layers.Dropout(0.2)(pooled_output)
	logits = tf.keras.layers.Dense(num_labels, name="classifier")(x)

	return tf.keras.Model(text_input, logits)


def train_with_data(train_df, val_df, max_epochs=None, batch_size_override=None):
	tf.random.set_seed(seed)

	epochs = max_epochs or MAX_EPOCHS
	train_ds, val_ds, num_labels, label_lookup = build_datasets_from_frames(
		train_df,
		val_df,
		batch_size_override=batch_size_override,
	)
	model = build_model(num_labels)

	steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
	total_steps = steps_per_epoch * epochs
	warmup_steps = int(0.1 * total_steps)

	optimizer = optimization.create_optimizer(
		init_lr=2e-5,
		num_train_steps=total_steps,
		num_warmup_steps=warmup_steps,
		optimizer_type="adamw",
	)

	model.compile(
		optimizer=optimizer,
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
	)

	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=epochs,
	)

	plt.figure(figsize=(8, 5))
	plt.plot(history.history.get("loss", []), label="train_loss")
	plt.plot(history.history.get("val_loss", []), label="val_loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Training Loss")
	plt.legend()
	plt.tight_layout()
	plt.show()

	return model, history, label_lookup