#ifndef __VID_ANALYZE_H__
#define __VID_ANALYZE_H__

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
#include "cache.h"
#include "audio.h"

using namespace cv;
using namespace std;
using json = nlohmann::json;

void analyzeScene(sourceT *s, int idx);
void detectScenes(sourceT *s, float lookback_seconds=2.0, float z_threshold=4.0);
void createDerez(sourceT *s, const char *derez_filename, float shrink=0.15);

void drawPaletteBox(sceneT *s, Mat screen, int x, int y, int sz, bool horizontal);
void drawMotionBox(sceneT *s, Mat screen, int x, int y, int sz, bool horizontal);

#endif 
