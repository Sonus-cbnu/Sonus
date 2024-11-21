# Sonus - Orchestra Instrument Sound Extraction Project

## Project Link

[Sonus Notion Link](https://reo91004.notion.site/sonus-2024?pvs=4)

## Introduction

Sonus is a project that utilizes AI technology to extract individual orchestra instrument sounds. This system provides separated sounds of each instrument, allowing musicians to practice orchestral pieces solo. By enhancing the practice environment and offering a variety of instrument sound samples, it aims to support learning and performance.

### Key Features

-   **Individual Instrument Sound Extraction**: Provides separated sounds of specific instruments to aid in practicing orchestral pieces individually.
-   **Extensive Sample Collection**: Offers a wide range of orchestra instrument samples useful for practice and education.
-   **Educational Use**: Aims to enable users to compare their performances with extracted instrument sounds.

## Project Goals

​<light>The main goal of this project is to efficiently extract individual instrument sounds from an orchestra.</light>​ This technology aims to:

-   Improve the environment for musicians to practice individually
-   Enhance the quality of music education by providing sounds of various instruments

## Dataset

The project utilizes the following datasets:

-   **Philharmonic Orchestra Dataset**
-   **University of Rochester Multi-Modal Music Performance Dataset**
-   **MusicNet Dataset**
-   Collects and uses data from **16 different orchestra instruments**.

## Methodology

### Feature-Based Instrument Identification

1. **Data Collection and Preprocessing**: Collected diverse sound data that includes individual instrument sounds, and performed audio preprocessing using MFCC.
2. **Data Augmentation**: Applied techniques such as Additive Noise Scaling, Frequency Masking, and Time Masking for data augmentation.
3. **Data Storage**: Used HDF5 format for efficient storage and divided data into training, validation, and test sets.
4. **CNN Model Design**: Designed a Convolutional Neural Network (CNN) to enhance the accuracy of sound extraction.

### Note-Based Instrument Identification

-   **Note-Level Instrument Assignment**: Applied a method of assigning each note event to an individual instrument class.
-   **Utilization of Note Information**: Used note data including pitch, start, and end information to classify instrument sounds.
-   **Modular Framework**: Designed as a modular framework that can integrate with the latest multi-pitch estimation (MPE) algorithms.
-   **Musically Synchronized CNN Architecture**: Adopted a CNN architecture using various kernel shapes (vertical, horizontal, square) to improve the accuracy of instrument assignment.

## Future Work

-   **Enhance Note-Level Instrument Classifier Performance**
-   **Develop a program that compares user performances with extracted instrument sounds**
-   **Build an educational system to improve the quality of practice and education**

## References

-   Lordelo et al., "Deep Learning Methods for Instrument Separation and Recognition" (2020b, 2021a)
-   Philharmonic Orchestra Dataset
-   University of Rochester Multi-Modal Music Performance Dataset
-   MusicNet Dataset

## Contributors

-   Yuntae Kim, Yongseong Park, Junhyuk Lee

## License

This project follows an open-source license.
