# Driver Fatigue Detection from EEG
This project was part of an university research project at the University of Technology Sydney 2019. Objective was to develop an algorithm detecting driver fatigue in a live EEG data stream. The detailed project writeup with all results obtained from the project is also contained in the repository. Only the feature extraction in MATLAB is published here. All users are encouraged to expand the feature extraction shown here or come up with an improved driver fatigue detection method based on these features.

This README will guide a user through the whole MATLAB setup so the program can run on your local machine.


## Getting Started

Since supervised learning methods are applied to solve the detection task, a labelled dataset is required. Such a dataset was already recorded and published here: [Hu and Min, 2018](https://figshare.com/articles/The_original_EEG_data_for_driver_fatigue_detection/5202739?fbclid=IwAR2pnHmmL58WffvA8hUESoxZPir4yawC7-ZIEQXHUqif4a6jDS0GfB82WlA). 

It is required to download all folders, named simply '1.zip' to '11.zip', and extract them into a folder on your local machine. 
### Pre-processing of all EEG Channels
Open *CNTpreprocess.m* and set the path variable to the dictionary with the EEG data. This script opens the raw EEG time signal from the download, splits it in custom detection epochs (f.e. 1 second) and assigns the labels (0 normal, 1 fatigued). An Bandpassfilter with a passband from 0.5 Hz up to 50 Hz is applied to the time signal. Thus, high frequency noise and DC offsets are removed from the signal without loosing relevant EEG signals for the fatigue detection. The user can select the EEG channel indices given in *Channelinfo.xls* that are going to be extracted here. Run the script to save the raw data in a folder */AllChannelsData* in your current working dictionary. All files containing the raw training data here should appear in mat-files *TrainingDataCh...*, seperated by their individual channel index.

### Feature Extraction
Next, this pre-processed training data is analyzed and the features for the supervised learning and testing are extracted by running *FeatureExtraction.m*. Driver fatigue can be observed by careful statistical analysis of the individual EEG brainwave channels alpha, beta, gamma and theta. More information on the feature selection is described in the project writeup contained in the repository. 

The selected features for the training dataset are:
1) Frequency spectra of alpha and beta channel
2) Standard deviation and mean of alpha beta time signal
3) Sample Entropy
4) Spectral Entropy
5) Khushaba Fuzzy Wavelet Packet Transform (WPT) feature extraction

All extracted features are saved in the mat-files */AllChannelsData/FeaturesCh...*, along with the previously extracted raw training data. Again, the files are seperated by their corresponding channel index.

### Supervised Learning 
All labelled features in the last step can now be fed into arbitrary supervised learning systems to train and test the detection algorithm. The detection algorithm proposed in the project is implemented in Knyme and explicit trained models are not listed this repository.


## Hints
Also read the comments in the code and inspect the generated data files to get an idea of the training data formats. This will help understanding the whole procedure.


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Thomas Wengerter** - *Initial work* - [thomaswengerter](https://github.com/thomaswengerter)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
