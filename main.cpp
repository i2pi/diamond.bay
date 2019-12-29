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
  unsigned long   duration;
  Mat             key_frame;
  vector<Point2f> *motion;
  Mat             palette;
  vector<float>   palette_weight;
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

  VideoCapture  *derez_cap;
} sourceT;

cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
VideoWriter   output_video;

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

void playSource (sourceT *s) {
  Mat             frame;
  unsigned long   i;

  printf ("Playing %s\n", s->filename);
    
  namedWindow( "play", 1 );

  s->raw_cap->set(CAP_PROP_POS_FRAMES, 0);

  for (i=0; i<s->frames; i++) {
    s->raw_cap->read(frame);
    if( frame.empty() ) break;

    imshow("play", frame);

    char c = (char)waitKey(1); 
    if (c != -1) break;
  }

  printf ("Done\n");
}

void showFrame(Mat frame, const char *title, float pct=-1.0, Mat mask = Mat()) {
  Mat screen;

  frame.copyTo(screen);
  if (screen.type() == CV_8UC1) {
    cvtColor(screen, screen, COLOR_GRAY2BGR);
  }
  ft2->putText(screen, title, Point( 50, 0 ), 100, Scalar(255,255,255), -1, LINE_AA,  false );
  if (pct >= 0.0) {
    line(screen, Point(0, 0), Point(screen.cols*pct, 0), Scalar(255,255,255), 10, LINE_AA); 
  }

  if (mask.rows == screen.rows) {
    screen += mask;
  }

  output_video.write(screen);
  imshow("diamond bay", screen);
  int c = waitKey(1);
  if (c != -1) {
    printf("Key pressed. Exiting.\n");
    exit(0);
  }
}



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
    for (int k=0; k<scene->palette_weight.size(); k++) {
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
      json j = (*s->cache)["scenes"];
      for (i=0; i<j.size(); i++) {
        scene.start_frame_num = j[i]["start_frame_num"];
        scene.duration= j[i]["duration"];
        s->derez_cap->set(CAP_PROP_POS_FRAMES, scene.start_frame_num + (scene.duration >> 1)); 
        s->derez_cap->read(scene.key_frame);

        json pal = j[i]["palette"];
        scene.palette = Mat(pal.size(), 3, CV_8U);
        for (int k=0; k<pal.size(); k++) {
          unsigned long c = strtol(pal[k].get<std::string>().c_str(), NULL, 0);
          scene.palette.at<Vec3b>(k)[0] = c & 0x0000FF;
          scene.palette.at<Vec3b>(k)[1] = c & 0x00FF00;
          scene.palette.at<Vec3b>(k)[2] = c & 0xFF0000;
        }

        json pal_w = j[i]["palette_weight"];
        for (int k=0; k<pal_w.size(); k++) {
          float w = pal_w[k];
          scene.palette_weight.push_back(w);
        }

        scene.motion = new vector<Point2f>();
        json mot = j[i]["motion"];
        for (int k=0; k<mot.size(); k++) {
          scene.motion->push_back(Point(mot[k]["x"], mot[k]["y"]));
        }

        s->scenes->push_back(scene);
      }
      printf("Found %ld scenes in cache\n", i);
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

VideoWriter startOutput(sourceT *s, int width=-1, int height=-1) {
  int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
  char outfile[128];

  snprintf(outfile, 120, "output.%ld.avi", time(NULL));
  printf("Writing output to %s\n", outfile);

  if (width == -1) {
    width = s->width;
    height = s->height;
  }

  VideoWriter video(outfile,codec, s->fps, Size(width,height));

  namedWindow("diamond bay", 1);

  return(video);
}



sourceT *openSource (const char *filename) {
  sourceT *s;
  char    derez_filename[1024];

  s = (sourceT *) malloc (sizeof(sourceT));
  if (!s) {
    fprintf (stderr, "Failed to alloc source\n");
    exit (-1);
  }

  s->filename = strdup(filename);

  s->raw_cap = new VideoCapture();
  s->raw_cap->open(s->filename);

  if (!s->raw_cap->isOpened()) {
    fprintf (stderr, "Failed to open %s\n", s->filename);
    exit(-1);
  }

  s->width = s->raw_cap->get(CAP_PROP_FRAME_WIDTH);
  s->height = s->raw_cap->get(CAP_PROP_FRAME_HEIGHT);
  s->fps = s->raw_cap->get(CAP_PROP_FPS);
  s->frames = s->raw_cap->get(CAP_PROP_FRAME_COUNT);

  printf ("Opened %s: %d x %d @ %4.2ffps - %ld frames %6.4fs\n",
      s->filename, s->width, s->height, s->fps, s->frames, s->frames / s->fps);

  openCache(s);

  s->derez_cap = new VideoCapture();
  snprintf (derez_filename, 1000, "DEREZ_%s.avi", s->filename);
  if (access(derez_filename, F_OK) == -1) {
    createDerez(s, derez_filename);
  }
  s->derez_cap->open(derez_filename);
  if (!s->derez_cap->isOpened()) {
    fprintf (stderr, "Failed to open %s\n", derez_filename);
    exit(-1);
  }

  s->motion_sample_points = new vector<Point2f>();

  int N = 5;
  int W = s->derez_cap->get(CAP_PROP_FRAME_WIDTH);
  int H = s->derez_cap->get(CAP_PROP_FRAME_HEIGHT);
  for (int x = W/(N+1); x <= W-W/(N+1); x += W/(N+1))
    for (int y = H/(N+1); y <= H-H/(N+1); y += H/(N+1)) {
    s->motion_sample_points->push_back(Point(x, y));
  }

  output_video = startOutput(s, 8192/4, 800);
  detectScenes(s);

  saveCache(s);

  return (s);
}

void bars (float x, int width=10, char c='*') {
  int i;
  int w = width * x;
  if (w > width) w = width;

  for (i=0; i<w; i++) {
    printf ("%c", c);
  }
  for (; i<width; i++) {
    printf(".");
  }
}

void analyzeMusic(const char *ogg_name, bool try_real_time = true) {
  PaStream  *stream;
  audioFileT    *music;
  int       p_idx;
  double    sec;
  int       frame, p_frame;
  int       lookback = 8192;
  int height = 800;
  int       master_idx;
  float mean = 0;

  music = read_ogg(ogg_name);
  stream = init_portaudio(2, 44100, music);

  sec = 0.0;
  p_frame = 0;
  p_idx = -1;
  frame = 0;
  Mat screen = Mat(height, lookback/4, CV_8UC3);
  Mat screen_blur = Mat(height, lookback/4, CV_8UC3);
  screen *= 0;
  for (master_idx = 0; master_idx < music->samples; master_idx++) {
    int idx;
    if (!try_real_time) {
      idx = master_idx;
    } else {
      idx = music->idx;
      master_idx = 0;
    }
    if (idx != p_idx) {
      bool new_frame = false;

      sec = idx / (2.0 * 44100.0);

      p_idx = idx;

      frame = sec * 29.97;

      if (frame > p_frame) {
        new_frame = true;

        if (idx > lookback) {
          vector<float> sample;
          Mat           FFT;

          for (int i=idx-lookback; i<idx; i++) {
            int j = i - (idx - lookback);
            float w = 0.5 * (1 - cos(2*PI*j/(float)lookback));
            sample.push_back(w*music->sample[i] / 32768.0);
          }
          dft(sample, FFT);

          blur(screen, screen, Size(50,5));
          screen *= 0.60;
          Mat rot_mat = getRotationMatrix2D( Point(mean, 400), 5.0*(mean-lookback/6)/200.0, 1.05);
          warpAffine(screen, screen_blur, rot_mat, screen.size() );
          screen = screen + 0.35*screen_blur;

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
          for (i=smooth; i<lookback/4-smooth; i++) {
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
            mean += i * sy[i];
            N += sy[i];
          }
          mean /= N;


          for (i=0; i<lookback/4; i++) {
            if (sy[i] > 1.15*ssy[i]) { 
                line (screen, Point(i, 0), Point(i, height), Scalar(64,255,64), 1);
            }
            line(screen, Point(i,height), Point(i,height*(1.0-y[i])), Scalar(0,0,0), 1);
            line(screen, Point(i,height), Point(i,height*(1.0-sy[i])), Scalar(65,255,255), 1);
            line(screen, Point(i,height), Point(i,height*(1.0-ssy[i])), Scalar(16,16,16), 1);
          }


          line (screen, Point(mean, 0), Point(mean, height), Scalar(0,65,255), 10);

          blur(screen, screen, Size(15,65));
          showFrame(screen, "audio");
        }


        p_frame = frame;
      }


      printf ("%8.4fs ", sec);
      printf ("%8d ", music->idx);
      printf ("%c", new_frame ? 'F' : ' ');
      printf ("%06d ", frame);
      printf ("\n");
    }
  }
}


int main( int argc, char** argv )
{
  sourceT *source;

  ft2->loadFontData( "futura1.ttf", 0 );
  source = openSource("diamondbay.clips.mov");
  analyzeMusic("05 sense_5days later.ogg", false);


 // source->raw_cap->release();
  destroyAllWindows();

  return 0;
}
