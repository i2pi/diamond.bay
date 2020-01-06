#ifndef __MAIN_H__
#define __MAIN_H__

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

#include "audio.h"

using namespace cv;
using namespace std;
using json = nlohmann::json;

#define PI 3.14159265354

typedef struct {
  unsigned long   start_frame_num;
  unsigned long   current_frame_num;
  unsigned long   duration;
  Mat             key_frame;
  vector<Point2f> *motion;
  Mat             palette;
  vector<float>   *palette_weight;
} sceneT;

typedef struct {
  json          *cache;

  char          *filename;
  VideoCapture  *raw_cap;
  int           width, height;
  unsigned long frames;
  double        fps;

  vector<Point2f> *motion_sample_points;
  vector<sceneT> *scenes;
  Mat             *scene_palette_distance;
  Mat             *scene_motion_distance;

  VideoCapture  *derez_cap;
} sourceT;

extern cv::Ptr<cv::freetype::FreeType2> ft2;

void showFrame(Mat frame, const char *title="", float pct=-1.0, Mat mask = Mat());

#endif // __MAIN_H__
