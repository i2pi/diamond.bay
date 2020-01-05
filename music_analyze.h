#ifndef __MUSIC_ANALYZE_H__
#define __MUSIC_ANALYZE_H__

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

void analyse_audio_frame(Mat screen, audioFileT *music, int idx, float *mean, float *peaks);
bool add_beat_to_rhythm(vector<int> *beat_idx, int idx, int *beat_idx_delta, float threshold = 0.02);

#endif
