# Documentación - Modelo de Difusión (DDPM)

## Arquitectura del Modelo (U-Net con Conexiones Skip)

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '14px' }}}%%
graph LR
    subgraph Inputs["Entradas"]
        I1["Imagen ruidosa\nx_t\n(28, 28, 1)"]
        I2["Timestep\nt\n(scalar)"]
    end

    subgraph TimestepEmb["Timestep Embedding"]
        E1["Embedding(200→32)"]
        E2["Dense(32→64, Swish)"]
        I2 --> E1 --> E2
    end

    subgraph Encoder["Encoder (Downsampling)"]
        C0["Conv2D(16, 3×3, same)\n(28, 28, 16)"]
        RB1["ResidualBlock(16)\n+ Timestep Inject\n(28, 28, 16)"]
        DS1["Conv2D(32, 3×3, stride=2)\n(14, 14, 32)"]
        RB2["ResidualBlock(32)\n+ Timestep Inject\n(14, 14, 32)"]
        DS2["Conv2D(64, 3×3, stride=2)\n(7, 7, 64)"]

        I1 --> C0 --> RB1 --> DS1 --> RB2 --> DS2
    end

    subgraph Bottleneck["Bottleneck"]
        RB3["ResidualBlock(64)\n+ Timestep Inject\n(7, 7, 64)"]
        DS2 --> RB3
    end

    subgraph Decoder["Decoder (Upsampling)"]
        UP1["UpSampling2D(2×)\n+ Conv2D(32, 3×3)\n(14, 14, 32)"]
        CAT1["Concatenate\n(14, 14, 64)"]
        RB4["ResidualBlock(32)\n+ Timestep Inject\n(14, 14, 32)"]
        UP2["UpSampling2D(2×)\n+ Conv2D(16, 3×3)\n(28, 28, 16)"]
        CAT2["Concatenate\n(28, 28, 32)"]
        RB5["ResidualBlock(16)\n+ Timestep Inject\n(28, 28, 16)"]

        RB3 --> UP1 --> CAT1 --> RB4 --> UP2 --> CAT2 --> RB5
    end

    subgraph SkipConns["Conexiones Skip"]
        RB1 -.->|"skip1 (28, 28, 16)"| CAT2
        RB2 -.->|"skip2 (14, 14, 32)"| CAT1
    end

    subgraph OutputHead["Salida"]
        GN["GroupNorm(4)"]
        SW["Swish Activation"]
        CO["Conv2D(1, 3×3, same)\nkernel= zeros\n(28, 28, 1)"]
        O1["Ruido predicho\nε_θ(x_t, t)\n(28, 28, 1)"]

        RB5 --> GN --> SW --> CO --> O1
    end

    E2 -->|"+ (adición)"| RB1
    E2 -->|"+ (adición)"| RB2
    E2 -->|"+ (adición)"| RB3
    E2 -->|"+ (adición)"| RB4
    E2 -->|"+ (adición)"| RB5
```

### Detalle de un ResidualBlock

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '14px' }}}%%
graph LR
    X["Entrada\nx (C_in)"]
    T["Timestep\nEmbedding (64)"]

    subgraph Branch1["Rama Principal"]
        C1["Conv2D(C_out, 3×3)\n(no bias)"]
        GN1["GroupNorm(4)"]
        S1["Swish"]
        TD["Dense(C_out, Swish)\n→ Reshape(1,1,C_out)"]
        AD1["Add"]
        C2["Conv2D(C_out, 3×3)\n(no bias)"]
        GN2["GroupNorm(4)"]
        S2["Swish"]

        X --> C1 --> GN1 --> S1 --> AD1 --> C2 --> GN2 --> S2
        T --> TD --> AD1
    end

    subgraph Branch2["Rama Residual"]
        COND{{"¿C_in ≠ C_out?"}}
        CV1["Conv2D(C_out, 1×1)"]
    end

    X --> COND
    COND -->|"No (skip directo)"| AD2["Add"]
    COND -->|"Sí (proyección)"| CV1 --> AD2

    S2 --> AD2
    AD2 --> Y["Salida\n(C_out)"]
```

---

## Flujo de Entrenamiento

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '14px' }}}%%
flowchart TD
    A["Dataset: Fashion-MNIST\n(60000, 28, 28, 1)\nNormalizado a [-1, 1]"] --> B["Batch size = 128\nShuffle buffer = 60000"]

    B --> C{"¿Quedan batches?\n(469 por epoch)"}

    C -->|"Sí"| D["t ~ Uniform(0, T=200)\nnoise ~ N(0, I)\n(mismas dimensiones que x₀)"]
    C -->|"No (fin de epoch)"| K["Generar 25 imágenes\nde muestra (fijas)\ncon ruido fijo seed=42"]

    D --> E["Proceso directo q_sample:\nx_t = √(ᾱ_t) · x₀ + √(1 - ᾱ_t) · noise\n( imagen ruidosa a tiempo t )"]

    E --> F["Forward pass UNet:\nnoise_pred = model([x_t, t])"]

    F --> G["Loss = MSE(noise, noise_pred)\n(Predicción de ruido ε)"]

    G --> H["GradientTape\nCalcular gradientes"]

    H --> I["Gradient Clipping\nGlobal norm → 1.0"]

    I --> J["Adam optimizer\nlr = 1e-4\nActualizar pesos"]

    J --> C

    K --> L["Guardar grid 5×5\nen diffusion_training_images/epoch_NNNN.png"]

    L --> M{"¿Quedan epochs?\n(100 totales)"}
    M -->|"Sí"| B
    M -->|"No"| N["plot_training_history\nGuardar loss curve\n→ diffusion_loss.png"]
    N --> O["Fin del entrenamiento"]
```

---

## Flujo de Muestreo (Inferencia / Reverse Process)

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '14px' }}}%%
flowchart TD
    A["z_T ~ N(0, I)\nRuido puro inicial\n(n_samples, 28, 28, 1)"] --> B["t = T-1 = 199"]

    B --> C{"t ≥ 0?"}

    C -->|"Sí"| D["noise_pred = UNet([x_t, t])"]

    D --> E["Posterior Mean:\nx̂ = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · noise_pred)"]

    E --> F{"t > 0?"}

    F -->|"Sí"| G["posterior_var = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)\nz_t ~ N(0, I) (fijo, pre-calculado)\nx_{t-1} = x̂ + √(posterior_var) · z_t"]
    F -->|"t = 0 (último paso)"| H["x₀ = x̂  (sin ruido añadido)"]

    G --> I["t = t - 1"]
    I --> C

    H --> J["Clip a [-1, 1]\nRescalar a [0, 1]"]

    J --> K["Guardar imagen generada\n(28, 28, 1)"]
```

---

## Parámetros Clave

| Parámetro | Valor |
|---|---|
| Timesteps (T) | 200 |
| Beta schedule | Lineal, 1e-4 → 0.02 |
| Batch size | 128 |
| Learning rate | 1e-4 (Adam) |
| Gradient clipping | Global norm 1.0 |
| Epochs | 100 |
| Loss | MSE (ε-prediction, peso uniforme) |
| Dataset | Fashion-MNIST (60K imágenes, 28×28×1) |
| Muestra | 25 imágenes (grid 5×5) por epoch |
