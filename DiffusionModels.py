import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time

# ======================
# DATA
# ======================
(x_train, _), _ = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 127.5 - 1.0   # normalización [-1,1]
x_train = np.expand_dims(x_train, -1)

batch_size = 128
dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(60000)
    .batch(batch_size)
)

# ======================
# DIFFUSION SCHEDULE
# ======================
T = 200  # 1000 es innecesario para MNIST

beta_start = 1e-4
beta_end = 0.02

betas = tf.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alpha_bar = tf.math.cumprod(alphas)

# ======================
# FORWARD PROCESS
# ======================
def q_sample(x0, t, noise):
    sqrt_alpha_bar = tf.sqrt(tf.gather(alpha_bar, t))
    sqrt_one_minus_alpha_bar = tf.sqrt(1 - tf.gather(alpha_bar, t))

    sqrt_alpha_bar = tf.reshape(sqrt_alpha_bar, (-1,1,1,1))
    sqrt_one_minus_alpha_bar = tf.reshape(sqrt_one_minus_alpha_bar, (-1,1,1,1))

    return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

def residual_block(x, t_embed, filters):
    residual = x

    # Primera convolución
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.GroupNormalization(
        groups=4
    )(x)

    x = layers.Activation("swish")(x)

    # Adaptar el embedding temporal
    t = layers.Dense(
        filters,
        activation="swish"
    )(t_embed)

    t = layers.Reshape(
        (1, 1, filters)
    )(t)

    # Incorporar el timestep
    x = layers.Add()([x, t])

    # Segunda convolución
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.GroupNormalization(
        groups=4
    )(x)

    x = layers.Activation("swish")(x)

    # Si cambia la cantidad de canales,
    # adaptar la conexión residual
    if residual.shape[-1] != filters:
        residual = layers.Conv2D(
            filters,
            kernel_size=1,
            padding="same"
        )(residual)

    return layers.Add()([x, residual])


def build_model():
    # =========================
    # ENTRADAS
    # =========================
    image_input = layers.Input(
        shape=(28, 28, 1),
        name="noisy_image"
    )

    time_input = layers.Input(
        shape=(),
        dtype=tf.int32,
        name="timestep"
    )

    # =========================
    # EMBEDDING TEMPORAL
    # =========================
    t_embed = layers.Embedding(
        input_dim=T,
        output_dim=32
    )(time_input)

    t_embed = layers.Dense(
        64,
        activation="swish"
    )(t_embed)

    # =========================
    # ENTRADA DE LA IMAGEN
    # =========================
    x = layers.Conv2D(
        16,
        kernel_size=3,
        padding="same"
    )(image_input)

    # Forma: 28x28x16

    # =========================
    # ENCODER
    # =========================
    skip1 = residual_block(
        x,
        t_embed,
        filters=16
    )

    # Forma: 28x28x16

    x = layers.Conv2D(
        32,
        kernel_size=3,
        strides=2,
        padding="same"
    )(skip1)

    # Forma: 14x14x32

    skip2 = residual_block(
        x,
        t_embed,
        filters=32
    )

    # Forma: 14x14x32

    x = layers.Conv2D(
        64,
        kernel_size=3,
        strides=2,
        padding="same"
    )(skip2)

    # Forma: 7x7x64

    # =========================
    # CUELLO DE BOTELLA
    # =========================
    x = residual_block(
        x,
        t_embed,
        filters=64
    )

    # Forma: 7x7x64

    # =========================
    # DECODER
    # =========================
    x = layers.UpSampling2D(
        size=2,
        interpolation="nearest"
    )(x)

    # Forma: 14x14x64

    x = layers.Conv2D(
        32,
        kernel_size=3,
        padding="same"
    )(x)

    # Forma: 14x14x32

    x = layers.Concatenate()([
        x,
        skip2
    ])

    # Forma: 14x14x64

    x = residual_block(
        x,
        t_embed,
        filters=32
    )

    # Forma: 14x14x32

    x = layers.UpSampling2D(
        size=2,
        interpolation="nearest"
    )(x)

    # Forma: 28x28x32

    x = layers.Conv2D(
        16,
        kernel_size=3,
        padding="same"
    )(x)

    # Forma: 28x28x16

    x = layers.Concatenate()([
        x,
        skip1
    ])

    # Forma: 28x28x32

    x = residual_block(
        x,
        t_embed,
        filters=16
    )

    # Forma: 28x28x16

    # =========================
    # SALIDA
    # =========================
    x = layers.GroupNormalization(
        groups=4
    )(x)

    x = layers.Activation("swish")(x)

    output = layers.Conv2D(
        1,
        kernel_size=3,
        padding="same",
        kernel_initializer="zeros"
    )(x)

    # Forma: 28x28x1
    # Representa el ruido predicho

    return tf.keras.Model(
        inputs=[image_input, time_input],
        outputs=output,
        name="SmallDiffusionUNet"
    )

model = build_model()
optimizer = tf.keras.optimizers.Adam(1e-4)

def sample_batch(
    model,
    initial_noise,
    reverse_noises
):

    x = tf.identity(initial_noise)

    for t in reversed(range(T)):
        t_tensor = tf.fill(
            [tf.shape(x)[0]],
            t
        )

        noise_pred = model(
            [x, t_tensor],
            training=False
        )

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]

        # Media del proceso inverso:
        # estimación de x_(t-1)
        x = (
            1.0 / tf.sqrt(alpha_t)
        ) * (
            x
            - (
                beta_t
                / tf.sqrt(1.0 - alpha_bar_t)
            ) * noise_pred
        )

        if t > 0:
            alpha_bar_previous = alpha_bar[t - 1]

            posterior_variance = (
                beta_t
                * (1.0 - alpha_bar_previous)
                / (1.0 - alpha_bar_t)
            )

            # Se usa el ruido fijo correspondiente a este paso.
            x += (
                tf.sqrt(posterior_variance)
                * reverse_noises[t]
            )

    return x

# =========================
# GUARDAR IMÁGENES
# =========================
def generate_and_save_images(
    model,
    epoch,
    initial_noise,
    reverse_noises,
    output_dir="diffusion_training_images"
):
    os.makedirs(output_dir, exist_ok=True)

    generated_images = sample_batch(
        model=model,
        initial_noise=initial_noise,
        reverse_noises=reverse_noises
    )

    # Las imágenes del dataset están normalizadas en [-1, 1].
    generated_images = tf.clip_by_value(
        generated_images,
        -1.0,
        1.0
    )

    # Transformar de [-1, 1] a [0, 1]
    generated_images = (
        generated_images + 1.0
    ) / 2.0

    fig = plt.figure(figsize=(10, 10))

    for i in range(generated_images.shape[0]):
        plt.subplot(5, 5, i + 1)

        image = generated_images[i, :, :, 0]

        plt.imshow(
            image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0
        )

        plt.axis("off")

    plt.suptitle(
        f"Época {epoch}",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"epoch_{epoch:04d}.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

# =========================
# TRAIN STEP
# =========================
@tf.function
def train_step(x0):
    batch_size = tf.shape(x0)[0]

    # Un timestep aleatorio para cada imagen.
    t = tf.random.uniform(
        shape=(batch_size,),
        minval=0,
        maxval=T,
        dtype=tf.int32
    )

    # Ruido real que debe aprender a predecir.
    noise = tf.random.normal(
        shape=tf.shape(x0)
    )

    # Imagen ruidosa correspondiente al timestep.
    x_t = q_sample(
        x0=x0,
        t=t,
        noise=noise
    )

    with tf.GradientTape() as tape:
        noise_pred = model(
            [x_t, t],
            training=True
        )

        loss = tf.reduce_mean(
            tf.square(
                noise - noise_pred
            )
        )

    grads = tape.gradient(
        loss,
        model.trainable_variables
    )

    grads, _ = tf.clip_by_global_norm(
        grads,
        1.0
    )

    optimizer.apply_gradients(
        zip(
            grads,
            model.trainable_variables
        )
    )

    return loss

# =========================
# TRAIN
# =========================
def train(
    dataset,
    epochs,
    n_samples=25
):
    loss_history = []
    loss_std_history = []

    # Ruido inicial fijo.
    # Todas las épocas comienzan desde exactamente estas
    # mismas 25 imágenes de ruido.
    fixed_initial_noise = tf.random.stateless_normal(
        shape=(n_samples, 28, 28, 1),
        seed=(42, 0)
    )

    # Ruido fijo para cada paso del proceso inverso.
    #
    # Forma:
    # (T, n_samples, 28, 28, 1)
    fixed_reverse_noises = tf.random.stateless_normal(
        shape=(T, n_samples, 28, 28, 1),
        seed=(42, 1)
    )

    for epoch in range(epochs):
        start = time.time()

        batch_losses = []

        for batch in dataset:
            loss = train_step(batch)

            batch_losses.append(
                float(loss.numpy())
            )

        loss_mean = np.mean(batch_losses)
        loss_std = np.std(batch_losses)

        loss_history.append(loss_mean)
        loss_std_history.append(loss_std)

        print(
            f"Época {epoch + 1}/{epochs}\n"
            f"Pérdida de ruido -> "
            f"Media: {loss_mean:.6f} | "
            f"Desv: {loss_std:.6f}\n"
            f"Tiempo: "
            f"{time.time() - start:.2f}s\n"
        )

        generate_and_save_images(
            model=model,
            epoch=epoch + 1,
            initial_noise=fixed_initial_noise,
            reverse_noises=fixed_reverse_noises
        )

    return {
        "loss": loss_history,
        "loss_std": loss_std_history
    }

history = train(
    dataset=dataset,
    epochs=100,
    n_samples=25
)

# =========================
# GRÁFICO DE PÉRDIDA
# =========================
def plot_training_history(
    history,
    output_path="diffusion_loss.png"
):
    epochs = np.arange(
        1,
        len(history["loss"]) + 1
    )

    mean_loss = np.array(
        history["loss"]
    )

    std_loss = np.array(
        history["loss_std"]
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        mean_loss,
        label="Pérdida media"
    )

    # Región media ± desviación estándar.
    #
    # Se evita que el límite inferior sea cero o negativo,
    # porque la escala logarítmica no admite esos valores.
    lower_bound = np.maximum(
        mean_loss - std_loss,
        1e-8
    )

    upper_bound = (
        mean_loss + std_loss
    )

    plt.fill_between(
        epochs,
        lower_bound,
        upper_bound,
        alpha=0.2,
        label="Media ± desviación"
    )

    plt.yscale("log")

    plt.xlabel("Época")
    plt.ylabel(
        "Pérdida media — escala logarítmica"
    )

    plt.title(
        "Evolución del entrenamiento "
        "del modelo de difusión"
    )

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.5
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    
plot_training_history(history)