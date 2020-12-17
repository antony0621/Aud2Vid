# Aud2Vid: Future Prediction Conditioned by Audio

***1. Optical flow estimation given audio and initial frames***

**2. TBD**



## dataset

**[MUSIC](https://github.com/roudimit/MUSIC_dataset)**

* ***Download:*** MUSIC21_solo_videos contains **21** kinds of instruments. All the downloaded videos have been trimmed into **3-second** clips(total number is 65480). After filtering those whose fps is less than 30, the dataset contains **40651** videos. 

  Acoustic_guitar 3608     Tuba 1090                Violin 2469                   Bagpipe 2721

  Saxophone 544              Piano 2985               Trumpet 572                Congas 3016

  Erhu 1071                       Guzheng 2947          Basson 2511                Clarinet 678

  Accordion 1610              Cello 1437                 Electric_bass 1822

  Flute 910                         Drum 2858                Ukulele 2140

  Banjo 1811                      Pipa 2386                  Xylophone 1465

  We split the dataset into training set, which consists of **36480** clips, and test set, whcih contains **4171** clips. When training, the validation set is 10% random samples of training set.

* ***Extract frame and audio:*** Extract frames and audio with opencv and ffmpeg, respectively. Each frame-audio pair contains **90** frames and 3-second audio.



## Structure

TODO





