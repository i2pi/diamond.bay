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

#include "audio.h"

using namespace cv;
using namespace std;
using json = nlohmann::json;


void openCache(sourceT *s, const char *filename="cache.json") {
  ifstream infile;

  s->cache = new json();

  infile.open (filename, ios::in | ios::ate);
  if (infile.is_open()) {
   streampos size;
   char     *buf;

   size = infile.tellg();
   buf = new char[size];
   infile.seekg (0, ios::beg);
   infile.read (buf, size);
   infile.close();
    try {
      *s->cache = json::parse(buf);
    } catch (nlohmann::detail::parse_error) {
      cout << "Unable to read cache file\n";
    }
  } else {
    cout << "No cache file found\n";
  }
}

void saveCache(sourceT *s, const char *filename="cache.json") {
  ofstream outfile;

  outfile.open(filename, ios::out);
  outfile << s->cache->dump(4) << endl;
  outfile.close();
}

void cacheScenes(sourceT *s) {
  int i;

  (*s->cache)["scenes"] = json::array();

  for (i = 0; i<s->scenes->size(); i++) {
    sceneT *scene = &s->scenes->at(i);
    int k;
    json pal = json::array();
    for (k=0; k<scene->palette.rows; k++) {
      Vec3b col = scene->palette.at<Vec3b>(k);
      char str[256];
      snprintf(str, 200, "0x%02x%02x%02x", 
          col[2], col[1], col[0]);

      pal.push_back(str);
    }

    json pal_w = json::array();
    for (int k=0; k<scene->palette_weight->size(); k++) {
      pal_w.push_back(scene->palette_weight[k]);
    }

    json mot = json::array();
    for (k=0; k<scene->motion->size(); k++) {
      Point2f p = (*scene->motion)[k];
      mot.push_back({{"x", p.x}, {"y", p.y}});
    }

    json sj = { 
      {"start_frame_num",scene->start_frame_num},
      {"duration", scene->duration},
      {"palette", pal},
      {"palette_weight", pal_w},
      {"motion", mot}
    };
    (*s->cache)["scenes"].push_back(sj);
  }

  saveCache(s);
}

void loadScenesFromCache(sourceT *s) {
  unsigned long i;
  sceneT        scene;


  json j = (*s->cache)["scenes"];
  for (i=0; i<j.size(); i++) {
    scene.start_frame_num = j[i]["start_frame_num"];
    scene.current_frame_num = scene.start_frame_num;
    scene.duration= j[i]["duration"];
    s->derez_cap->set(CAP_PROP_POS_FRAMES, scene.start_frame_num + (scene.duration >> 1)); 
    s->derez_cap->read(scene.key_frame);

    json pal = j[i]["palette"];
    scene.palette = Mat(pal.size(), 3, CV_8U);
    for (int k=0; k<pal.size(); k++) {
      unsigned long c = strtol(pal[k].get<std::string>().c_str(), NULL, 0);
      scene.palette.at<Vec3b>(k)[0] = c & 0x0000FF;
      scene.palette.at<Vec3b>(k)[1] = (c & 0x00FF00) >> 8;
      scene.palette.at<Vec3b>(k)[2] = (c & 0xFF0000) >> 16;
    }

    json pal_w = j[i]["palette_weight"];
    scene.palette_weight = new vector<float>();
    for (int k=0; k<pal_w.size(); k++) {
      float w = pal_w[k];
      scene.palette_weight->push_back(w);
    }

    scene.motion = new vector<Point2f>();
    json mot = j[i]["motion"];
    for (int k=0; k<mot.size(); k++) {
      float x, y;
      x = mot[k]["x"];
      y = mot[k]["y"];
      scene.motion->push_back(Point2f(x, y));
    }

    if (pal.size() > 0) s->scenes->push_back(scene);
  }
  printf("Found %ld scenes in cache\n", s->scenes->size());
}
