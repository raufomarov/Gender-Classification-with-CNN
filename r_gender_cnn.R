library(keras)
library(tidyverse)

train_dir <- file.path("Gender",'Train')
validation_dir <- file.path("Gender","Validation")
test_dir <- file.path("Gender","Test")

## Normalize images by dividing all values to 255

train_generated_data <- image_data_generator(rescale = 1/255)
validation_generated_data <- image_data_generator(rescale = 1/255)
test_generated_data <- image_data_generator(rescale = 1/255)

### flow images from directory

train_generator <- flow_images_from_directory(
  train_dir,
  train_generated_data,
  target_size = c(128, 128),
  batch_size = 10,
  class_mode = "binary" 
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_generated_data,
  target_size = c(128, 128),
  batch_size = 10,
  class_mode = "binary"
  
)

### defining model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(128, 128, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c('acc'),
)

### fit model 

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 5,
  callbacks = callback_tensorboard("logs/run_a"),
  validation_data = validation_generator,
  validation_steps = 50,
  verbose = 2)

### call tensorboard

tensorboard("logs/run_a")


##### use model to evaluate test images

test_generator <- flow_images_from_directory(
  test_dir,
  test_generated_data,
  target_size = c(128, 128),
  batch_size = 20,
  class_mode = "binary"
)


model %>% evaluate_generator(test_generator, steps = 50)

