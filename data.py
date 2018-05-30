import os
import shutil


class Data(object):
    """
    Helper class for managing the data that is used in the task
    """

    raw_persons_location = './assets/persons/'
    """ The directory where the faces are located """

    gender_male_file = './assets/males.txt'
    """ The location of the text file with male names """

    gender_female_file = './assets/females.txt'
    """ The location of the text file with female names """

    gender_files_encoding = 'utf8'
    """ The encoding of the gender files """

    __files_temp_location = './tmp'
    """ Temporary file location for gender files """

    __files_train_folder = 'train'
    """ Temporary file location for training files """

    __files_test_folder = 'test'
    """ Temporary file location for test files """

    __male_list = []
    """ List of all the images of males """

    __female_list = []
    """ List off all images of females"""

    __verification_data_percent = 10
    """ How many percent of the dataset that is to be used as verification data """

    def __init__(self):
        super(Data, self).__init__()

    def ensure_tmp_is_created_and_structured(self):
        """
        Ensures that the tmp folder is created and fixed
        :return: None
        """
        if os.path.isdir(self.__files_temp_location):
            print(f"Tmp folder seems to be in place, no need to format the data")
            return

        # Create tmp directories
        self.__create_temp_directories()

        # Create gender folders
        self.format_gender_data()

        # Create person folders
        self.format_person_data()

    def format_gender_data(self):
        """
        Formats and prepares the data for gender recognition
        :return: None
        """
        print(f"Starting to preparation for gender data, please wait...")

        # Create file lists
        self.__load_gender_lists()

        # Create persons dict
        persons = dict()

        for path, dirs, files in os.walk(self.raw_persons_location):
            for file in files:
                persons[file] = f"{path}/{file}"

        # File number counter
        filenumber = 0

        for filename, filepath in persons.items():

            # Add one to file number
            filenumber += 1

            if filenumber % self.__verification_data_percent is 0:
                subfolder = self.__files_test_folder
            else:
                subfolder = self.__files_train_folder

            # Check if male
            if filename in self.__male_list:
                self.__copy_file_to_directory(filepath, f"{self.__files_temp_location}/genders/{subfolder}/male/{filename}")
                continue

            # Check if female
            if filename in self.__female_list:
                self.__copy_file_to_directory(filepath, f"{self.__files_temp_location}/genders/{subfolder}/female/{filename}")
                continue

            print(f"File: {filename} not in female or male list")

        print(f"Gender data is now ready")

    def format_person_data(self):
        """
        Formats and prepares the data for individual recognition
        :return: None
        """
        print(f"Starting to preparation for individual data, please wait...")

        # Create persons dict
        persons = dict()

        # Create list of all the persons
        for path, dirs, files in os.walk(self.raw_persons_location):
            if os.path.basename(os.path.normpath(path)) == "persons":
                continue

            persons[os.path.basename(os.path.normpath(path))] = [f"{path}/{file}" for file in files]

        # Start copying data into right location
        for person, files in persons.items():

            # Get the files for the person
            person_files = files

            # Create the required folders
            os.mkdir(f"{self.__files_temp_location}/individuals/test/{person}/")
            os.mkdir(f"{self.__files_temp_location}/individuals/train/{person}/")

            # Check if person has more than one image, so that we may use one of it to validate
            if len(files) > 1:
                # Copy the test image to test directory
                self.__copy_file_to_directory(person_files[0], f"{self.__files_temp_location}/individuals/test/{person}/")

                # Remove the copied element
                person_files = person_files[1:]

            # Copy each file to the train folder
            for file in person_files:
                self.__copy_file_to_directory(file, f"{self.__files_temp_location}/individuals/train/{person}/")

        print(f"Individual data is now ready")

    def cleanup_temp_folder(self):
        """
        Deletes the temp folder, use this when the program is done to cleanup used space
        :return:
        """
        if os.path.exists(self.__files_temp_location):
            shutil.rmtree(self.__files_temp_location)
            print(f"Directory '{self.__files_temp_location}' has been removed")

    def __load_gender_lists(self):
        """
        Loads the gender files into the local "database"
        :return: None
        """

        # Load the males
        with open(self.gender_male_file, 'r', encoding=self.gender_files_encoding) as file:
            for filename in file:
                self.__male_list.append(filename.strip())

        # Load the females
        with open(self.gender_female_file, 'r', encoding=self.gender_files_encoding) as file:
            for filename in file:
                self.__female_list.append(filename.strip())

    def __create_temp_directories(self):
        """
        Ensures that temp directories are created and in place
        :return: None
        """

        if os.path.exists(self.__files_temp_location):
            print(f"Directory '{self.__files_temp_location}' already exists, deleting it")
            shutil.rmtree(self.__files_temp_location)

        # Create main temp directory
        os.mkdir(self.__files_temp_location)

        # Create genders folders
        os.mkdir(f"{self.__files_temp_location}/genders")
        os.mkdir(f"{self.__files_temp_location}/genders/{self.__files_train_folder}")
        os.mkdir(f"{self.__files_temp_location}/genders/{self.__files_train_folder}/male")
        os.mkdir(f"{self.__files_temp_location}/genders/{self.__files_train_folder}/female")
        os.mkdir(f"{self.__files_temp_location}/genders/{self.__files_test_folder}")
        os.mkdir(f"{self.__files_temp_location}/genders/{self.__files_test_folder}/male")
        os.mkdir(f"{self.__files_temp_location}/genders/{self.__files_test_folder}/female")

        # Create individual folders
        os.mkdir(f"{self.__files_temp_location}/individuals")
        os.mkdir(f"{self.__files_temp_location}/individuals/{self.__files_train_folder}")
        os.mkdir(f"{self.__files_temp_location}/individuals/{self.__files_test_folder}")

    @staticmethod
    def __copy_file_to_directory(source, destination):
        """
        Helper method for copying files from one destination to another
        :param source:
        :param destination:
        :return: None
        """
        shutil.copy(source, destination)
