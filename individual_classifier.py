from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import backend as keras_backend
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


class IndividualClassifier(object):

    source_directory = './tmp/individuals/'
    """ The filepath to the source files that is pre-arranged in gender folders """

    batch_size = 100
    """ The batch size for training """

    epochs = 30
    """ Number of epochs to train """

    __image_size = (32, 32)
    """ The size of the images rows x cols """

    __input_shape = None
    """ The input format for the model """

    __number_of_classes = 5749
    """ The amount of different persons """

    __keras_model = None
    """ The keras model """

    __keras_training_data = None
    """ The training data used with Keras """

    __keras_test_data = None
    """ The test data used with Keras """

    def create_model(self):
        """
        Creates the Keras model
        :return: None
        """
        # Calculate the input shape
        self.__calculate_the_input_shape()

        # Find the amount of classes
        self.__get_the_amount_of_different_persons()

        self.__keras_model = Sequential()
        self.__keras_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        self.__keras_model.add(Conv2D(64, (3, 3), activation='relu'))
        self.__keras_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.__keras_model.add(Dropout(0.25))
        self.__keras_model.add(Flatten())
        self.__keras_model.add(Dense(128, activation='relu'))
        self.__keras_model.add(Dropout(0.5))
        self.__keras_model.add(Dense(self.__number_of_classes, activation='softmax'))

    def fit_model(self):
        """
        Fits the created model
        :return: None
        """

        # Compile the model for better performance
        self.__keras_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

        self.__keras_model.fit_generator(
            self.__keras_training_data,
            epochs=self.epochs,
            validation_data=self.__keras_test_data)

    def print_model_summary(self):
        """
        Prints a summary of the model
        :return: None
        """
        self.__keras_model.summary()

    def create_train_verification_data(self):
        """
        Creates training and validation data pairs
        :return: None
        """
        # Training data
        self.__keras_training_data = ImageDataGenerator().flow_from_directory(
            f"{self.source_directory}/train",
            target_size=self.__image_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        # Test data
        self.__keras_test_data = ImageDataGenerator().flow_from_directory(
            f"{self.source_directory}/test",
            target_size=self.__image_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )

    def __calculate_the_input_shape(self):
        """
        Calculate how the input model should look like
        :return: None
        """

        if keras_backend.image_data_format() == 'channels_first':
            self.input_shape = (3, self.__image_size[0], self.__image_size[1])
            return

        self.input_shape = (self.__image_size[0], self.__image_size[1], 3)
