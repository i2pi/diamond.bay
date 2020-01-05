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

class RunningStat {
    // From John Cook
    public:
        RunningStat() : m_n(0) {}

        void Clear() {
            m_n = 0;
        }

        void Push(double x) {
            m_n++;

            // See Knuth TAOCP vol 2, 3rd edition, page 232
            if (m_n == 1) {
                m_oldM = m_newM = x;
                m_oldS = 0.0;
            } else {
                m_newM = m_oldM + (x - m_oldM)/m_n;
                m_newS = m_oldS + (x - m_oldM)*(x - m_newM);
    
                // set up for next iteration
                m_oldM = m_newM; 
                m_oldS = m_newS;
            }
        }

        int NumDataValues() const {
            return m_n;
        }

        double Mean() const {
            return (m_n > 0) ? m_newM : 0.0;
        }

        double Variance() const {
            return ( (m_n > 1) ? m_newS/(m_n - 1) : 0.0 );
        }

        double StandardDeviation() const {
            return sqrt( Variance() );
        }

    private:
        int m_n;
        double m_oldM, m_newM, m_oldS, m_newS;
};

void analyzeScene(sourceT *s, int idx) {

  sceneT *scene;
  unsigned long i;
  Mat img;

  printf ("ANALYZING SCENE %d\n", idx);


  scene = &s->scenes->at(idx);
  s->derez_cap->set(CAP_PROP_POS_FRAMES, scene->start_frame_num + (scene->duration >> 1)); 
  s->derez_cap->read(scene->key_frame);

  int K = 6;

  Mat ocv; 
  scene->key_frame.copyTo(ocv);
  Mat data;
  cvtColor(ocv, ocv, COLOR_BGR2Lab);
  ocv.convertTo(data,CV_32F);
  data = data.reshape(1,data.total());

  // do kmeans
  Mat labels;
  kmeans(data, K, labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 1,
           KMEANS_PP_CENTERS, scene->palette);

  // reshape both to a single row of Vec3f pixels:
  scene->palette= scene->palette.reshape(3,scene->palette.rows);
  scene->palette.copyTo(scene->palette);

  // back to 2d, and uchar:
  scene->palette.convertTo(scene->palette, CV_8UC3);
  cvtColor(scene->palette, scene->palette, COLOR_Lab2BGR);

  scene->palette_weight = vector<float>(K);
  for (i=0; i<labels.rows; i++) {
      scene->palette_weight[labels.at<int>(i)] ++;
  } 
  for (i=0; i<K; i++) scene->palette_weight[i] /= (float) labels.rows;

  Mat frame = scene->key_frame;

  Mat gray, prevGray;
  s->derez_cap->set(CAP_PROP_POS_FRAMES, scene->start_frame_num);
  s->derez_cap->read(prevGray);
  cvtColor(prevGray, prevGray, COLOR_BGR2GRAY);
  Size winSize(50,50);

  vector<int> sampleCounts;
  scene->motion = new vector<Point2f>();
  for (i=0; i<s->motion_sample_points->size(); i++) {
    scene->motion->push_back(Point(0.0,0.0));
    sampleCounts.push_back(0);
  }


  for (i=1; i < scene->duration-3; i++) {
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    vector<uchar> status;
    vector<float> err;
    vector<Point2f> outPoints;
    Mat img;

    s->derez_cap->read(img); 
    cvtColor(img, img, COLOR_BGR2GRAY);
    cvtColor(img, img, COLOR_GRAY2BGR);
    cvtColor(img, gray, COLOR_BGR2GRAY);
    calcOpticalFlowPyrLK(prevGray, gray, *s->motion_sample_points, outPoints,
        status, err, winSize,
        3, termcrit, 0, 0.001); 
    gray.copyTo(prevGray);

    for (int j=0; j<s->motion_sample_points->size(); j++) {
      if (status[j]) { 
        Point2f motion = outPoints[j] - (*s->motion_sample_points)[j];
        sampleCounts.at(j)++;
        scene->motion->at(j) += motion;

        arrowedLine(img, (*s->motion_sample_points)[j], 
            (*s->motion_sample_points)[j] + motion * 10, Scalar(0,255,0), 2, LINE_AA, 0, 0.3);
      } else {
        drawMarker(img, (*s->motion_sample_points)[j], Scalar(0,255,0));
      }
    }
//    if (i % 20 == 0) showFrame(img, "motion", -1);
  }

  gray.copyTo(img);
  img *= 0.25;
  cvtColor(img, img, COLOR_GRAY2BGR);

  for (i=0; i<K; i++) {
    Point A, B; 
    int   x, X, y, Y;

    x = frame.cols / 2;
    X = frame.cols / 5;

    y = 50;
    Y = (frame.rows-100) / 3;

    if (i % 2 == 0) {
      A = Point(x,  y+Y*(i/2));
      B = Point(x-X, y+Y*(1+i/2));
    } else {
      A = Point(x, y+Y*(i/2));
      B = Point(x+X, y+Y*(1+i/2)); 
    }

    rectangle(img, A, B, scene->palette.at<Vec3b>(i), -1);
  }

  for (i=0; i<s->motion_sample_points->size(); i++) {
    drawMarker(img, (*s->motion_sample_points)[i], Scalar(128,128,128));
    if (sampleCounts.at(i) > 2) { 
      scene->motion->at(i) /= sampleCounts.at(i);
      arrowedLine(img, (*s->motion_sample_points)[i], 
          (*s->motion_sample_points)[i] + scene->motion->at(i) * 15, Scalar(255,255,255), 2, LINE_AA, 0, 0.3);
    }
  }
/*
  for (i=0; i<80; i++) 
      showFrame(img, "", -1);
*/
}

void detectScenes(sourceT *s, float lookback_seconds=2.0, float z_threshold=4.0) {
  Mat frame, p_frame, d_frame, mask;
  unsigned long i;
  unsigned long lookback_frames;
  int           scene_count = 0;
  RunningStat   rs;
  sceneT        scene, p_scene;

  s->scenes = new vector<sceneT>;

  if (((*s->cache)["scenes"]).is_array()) {
    cout << "Found scene cache\n";
    loadScenesFromCache(s);
    return;
  }

  lookback_frames = lookback_seconds * s->fps;

  s->derez_cap->set(CAP_PROP_POS_FRAMES, 0);
  s->derez_cap->read(p_frame);

  p_frame.copyTo(mask);
  mask = mask * 0;

  cvtColor(p_frame, p_frame, COLOR_BGR2GRAY);

  p_scene.start_frame_num = 0;
    
  for (i=1; i<s->frames; i++) {
    char  txt[256];
    double delta;
    Mat f, pf;

    s->derez_cap->read(frame);
    cvtColor(frame, frame, COLOR_BGR2GRAY);

    frame.convertTo(f, CV_32F);
    p_frame.convertTo(pf, CV_32F);

    d_frame = f - pf;

    delta = 0.0;
    for(int j = 0; j < d_frame.rows; j++) {
      const float *Mj = d_frame.ptr<float>(j);
      for(int k = 0; k < d_frame.cols; k++) { 
        delta += fabs(Mj[k]);
      }
    }
    delta /= ((double) 128.0 * d_frame.rows * d_frame.cols);


    if (rs.NumDataValues() >= lookback_frames) {
      double score;
      score = (delta - rs.Mean()) / rs.StandardDeviation();

      if (score >= z_threshold) {

        scene.start_frame_num = i;

        p_scene.duration = i - p_scene.start_frame_num;

        s->scenes->push_back(p_scene);
        analyzeScene(s, scene_count);

        p_scene = scene;

        scene_count++;
        printf ("%03d: Frame %05ld, Score = %f\n", scene_count, i, score);
        rs.Clear();
        line(mask, Point(mask.cols*i/s->frames, mask.rows), 
            Point(mask.cols*i/s->frames, mask.rows * (1.0 - score/z_threshold)), 
            Scalar(0,0,255), 1, LINE_AA);
      } else {
        line(mask, Point(mask.cols*i/s->frames, mask.rows), 
            Point(mask.cols*i/s->frames, mask.rows * (1.0 - score/z_threshold)), 
            Scalar(64,64,64), 1, LINE_AA);
      }
    } 

    rs.Push(delta);

    snprintf (txt, 200, "clip: %d", scene_count);
    if (i % 30 == 0) showFrame(frame, txt, -1, mask);

    frame.copyTo(p_frame);
  }
  p_scene.duration = i - p_scene.start_frame_num;
  s->scenes->push_back(p_scene);

  cacheScenes(s);

  printf ("Done detecting scenes\n");

}

void createDerez(sourceT *s, const char *derez_filename, float shrink=0.15) {
  unsigned long i;
  Mat img, d_img;
  int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
  Size derezSize;
  int pct, p_pct;

  derezSize = Size(s->width*shrink, s->height*shrink);
  VideoWriter video(derez_filename, codec, s->fps, derezSize);

  if (!video.isOpened()) {
    cerr << "Could not open the output video file for write\n";
    exit(-1);
  }

  s->raw_cap->set(CAP_PROP_POS_FRAMES, 0);

  p_pct = -1;
  for (i=0; i<s->frames; i++) {
    s->raw_cap->read(img);

    resize(img, d_img, derezSize);
    video.write(d_img);

    showFrame(d_img, "Derezing.", i / (float) s->frames);

    pct = 100*i / s->frames;
    if (pct > p_pct) {
      printf ("%d%% derezed\n", pct);
    }
    p_pct = pct;
  }

  video.release();
}
