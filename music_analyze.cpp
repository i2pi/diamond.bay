#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/freetype.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include <fstream>
#include <ctype.h>
#include <unistd.h>
#include <stdlib.h>

#include "json.hpp"

#include "diamond.h"
#include "vid_analyze.h"
#include "cache.h"
#include "audio.h"

using namespace cv;
using namespace std;

void analyse_audio_frame(Mat screen, audioFileT *music, int idx, float *mean, float *peaks) {
  int       lookback = 8192;
  int height = screen.rows;


  if (idx < lookback) {
    return;
  }

  vector<float> sample;
  Mat           FFT;

  for (int i=idx-lookback; i<idx; i++) {
    int j = i - (idx - lookback);
    float w = 0.5 * (1 - cos(2*PI*j/(float)lookback));
    sample.push_back(w*music->sample[i] / 32768.0);
  }
  dft(sample, FFT);

 const float *f = FFT.ptr<float>(0);
  float y[lookback];
  float sy[lookback];
  float ssy[lookback];
  int i;

  for (i=0; i<lookback/4; i++) {
    y[i] = sqrt(f[i]*f[i] + f[lookback-i-1]*f[lookback-i-1]);
    y[i] *= 0.1;
    y[i] = log(y[i] + 1.0);
  }

  for (i=0; i<100; i++) y[i] = 0;

  int smooth = 20;
  float N = 0;
  *mean = 0;
  for (i=smooth*2; i<lookback/4-smooth*2; i++) {
    float x, xx;
    x = 0;
    xx = 0;
    if (i > smooth*2) for (int j=i-smooth*2; j<i-smooth; j++ ) {
      xx += y[j] * 0.5;
    }
    for (int j=i-smooth; j<=i+smooth; j++) {
      float w = 0.5 * (1 - cos(2*PI*(j-i+smooth)/(float)(2*smooth+1)));
      x += w*y[j];
      xx += w*y[j];
    } 
    for (int j=i+smooth+1; j<i+smooth*2; j++) {
      xx += y[j] * 0.5;
    }
    sy[i] = x / (float) (2 * smooth + 1);
    ssy[i] = xx / (float) (4 * smooth + 1);
    *mean += i * sy[i];
    N += sy[i];

    if (sy[i] > 1.15*ssy[i]) {
      *peaks += 1.0;
    }
  }
  *mean /= N;
  *peaks /= lookback/4.0;

  Mat screen_blur = Mat(screen.rows, screen.cols, CV_8UC3);

  blur(screen, screen, Size(5,5));
  screen *= 0.40;

  Mat rot_mat = getRotationMatrix2D( Point(*mean, screen.rows/2), 5.0*(*mean-lookback/10)/200.0, 2.15);
  warpAffine(screen, screen_blur, rot_mat, screen.size() );
  screen = screen + 0.15*screen_blur;

  for (i=0; i<lookback/4; i++) {
    if (sy[i] > 1.15*ssy[i]) { 
      line (screen, Point(i, 0), Point(i, height), Scalar(64,255,64), 1);
    }
    int x = (i*screen.cols/ (lookback/4));
    line(screen, Point(x,height), Point(x,height*(1.0-y[i])), Scalar(0,0,0), 1);
    line(screen, Point(x,height), Point(x,height*(1.0-sy[i])), Scalar(65,255,255), 1);
    line(screen, Point(x,height), Point(x,height*(1.0-ssy[i])), Scalar(16,16,16), 1);
  }

  int mean_x = (*mean*screen.cols) / (lookback/4);

  line (screen, Point(mean_x, 0), Point(mean_x, height), Scalar(0,65,255), 10);
  blur(screen, screen, Size(5,5));
  line (screen, Point(mean_x, 0), Point(mean_x, height), Scalar(0,65,255), 10);

  *mean /= lookback/4.0;
}

bool add_beat_to_rhythm(vector<int> *beat_idx, int idx, int *beat_idx_delta, float threshold = 0.02) {
  if (beat_idx->size() < 2) {
    beat_idx->push_back(idx);
    return (true);
  } 

  float mean_delta = 0;
  for (int i=1; i<beat_idx->size(); i++) {
    mean_delta += beat_idx->at(i) - beat_idx->at(i-1);
  }
  mean_delta /= (float)(beat_idx->size() - 1.0);

  *beat_idx_delta = round(mean_delta);

  float delta = idx - beat_idx->back();

  float err = (delta - mean_delta) / mean_delta;

  if (delta > mean_delta*0.25) {
    int   multiple;
    err = fabs(err);
    multiple = (int)round(err);
    err = fabs(err - multiple);
/*
   float IPS = 44100 * 2.0;
    printf ("%4lu, %7.4f, %6.4f, %6.4f, %d, %+5.4f %5.4f", 
        beat_idx->size(), 
        idx / IPS, 
        delta / IPS, 
        mean_delta / IPS, 
        multiple,
        (delta - mean_delta) / mean_delta,
        err);
        */

    if (err < threshold) {
      beat_idx->push_back(idx - multiple*mean_delta);
      return (true);
    }
  }

  return (false);
}

