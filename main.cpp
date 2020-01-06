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
#include "music_analyze.h"
#include "cache.h"
#include "audio.h"

using namespace cv;
using namespace std;
using json = nlohmann::json;

cv::Ptr<cv::freetype::FreeType2> ft2;
VideoWriter   output_video;

void showFrame(Mat frame, const char *title, float pct, Mat mask) {
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

VideoWriter startOutput(sourceT *s, int width=-1, int height=-1) {
  int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
  char outfile[128];

  snprintf(outfile, 120, "output.%ld.avi", time(NULL));
  printf("Writing output to %s\n", outfile);

  if (width == -1) {
    width = s->width;
    height = s->height;
  }

  width = 800;
  height = 800;

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

  output_video = startOutput(s, W, H);
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

int kernText(Mat *screen, const char *text, float *kern, int height, Point loc=Point(0,0), Scalar col=Scalar(255,255,255)) {
  int baseline;
  Size sz;
  int i;
  int x;

  x = loc.x;

  for (i=0; i<strlen(text); i++) {
    char str[10];
    snprintf (str, 4, "%c", text[i]);
    if (screen != NULL) ft2->putText(*screen, str, Point(x, loc.y), height, col, -1, LINE_AA, false);
    sz = ft2->getTextSize(str, height, -1, &baseline);
    x += sz.width * kern[i];
  }

  return(x);

}

void title(Mat screen, sourceT *s) {
  int i;

  float kern[] = {0.95, 1.4, 1.0, 0.92, 0.95, 1.0, 1.0};
  float letterHeight = 500 * screen.rows / (float) s->height;
  float x, y;

  for (i=0; i<8; i++) kern[i] *= 1.1;
  x = screen.cols / 2;
  x -= kernText(NULL, "DIAMOND", kern, letterHeight)/2.0;
  y = 0;
  kernText(&screen, "DIAMOND", kern, letterHeight, Point(x,y));

  kern[0] = 1.0;
  kern[1] = 0.75;
  for (i=0; i<8; i++) kern[i] *= 1.1;
  x = screen.cols / 2;
  x -= kernText(NULL, "BAY", kern, letterHeight)/2.0;
  y = screen.rows - letterHeight * 1.3;
  kernText(&screen, "BAY", kern, letterHeight, Point(x,y));
}  


typedef struct {
  float val;
  int idx;
} orderT;

int orderCmp(const void *a, const void *b) {
  orderT *oa, *ob;
  oa = (orderT *) a;
  ob = (orderT *) b;

  if (oa->val < ob->val) return (-1);
  if (oa->val > ob->val) return (1);
  return(0);
}

int markovStep(Mat *distance_matrix, int i, float scale = 1.0, float p = 1.0) {
  int n = distance_matrix->rows;
  const float *d = distance_matrix->ptr<float>(i);
  orderT sd[1024];
  float ssd;
  float r;

  if (n > 1024) exit (-1);


  ssd = 0;
  for (int j=0; j<n; j++) {
    sd[j].val = pow(d[j], p) * scale;
    sd[j].idx = j;
    ssd += sd[j].val;
  } 
  for (int j=0; j<n; j++) sd[j].val /= fabs(ssd);

  qsort(&sd, n, sizeof(orderT), orderCmp);
  
  r = random() / (float)RAND_MAX;
  ssd = 0;
  int j = 0;
  while ((ssd < r) && (j < n)) ssd += fabs(sd[sd[j++].idx].val);

  return (sd[j-1].idx); 
}


void play(const char *mov_name, const char *ogg_name, bool try_real_time = true, bool derez=true)  {
  PaStream  *stream;
  audioFileT    *music;
  int       p_idx;
  int       frame, p_frame;
  int       master_idx;
  sourceT   *source;
  float sec, p_sec;
  float     audio_mean, audio_peaks;
  float     p_audio_peaks = 0;
  vector<int> beat_idx;
  bool      is_beat = false;
  int       current_scene_num;
  int       next_scene_num;
  sceneT    *current_scene;
  int       scene_start_frame = 0;
  int       last_beat_idx = 0;
  int       beat_idx_delta=RAND_MAX;

  srandom(time(NULL));

  source = openSource(mov_name);

  music = read_ogg(ogg_name);

  stream = init_portaudio(2, 44100, music);

  int width, height;

  if (derez) {
    width = source->derez_cap->get(CAP_PROP_FRAME_WIDTH);
    height = source->derez_cap->get(CAP_PROP_FRAME_HEIGHT);
  } else {
    width = source->width;
    height = source->height;
  }

  Mat screen = Mat(height, width, CV_8UC3);
  Mat mask = Mat(height, width, CV_8UC3);
  Mat anal = Mat(height, width, CV_8UC3);

  p_sec = 0;
  p_idx = -1;
  p_frame = -1;

  current_scene_num = 0;
  current_scene = &source->scenes->at(current_scene_num);
  scene_start_frame = 0;


  for (master_idx = 0; master_idx < music->samples; master_idx++) {
    int idx;
    
    if (try_real_time) {
      idx = music->idx;
      master_idx = 0;
    } else {
      idx = master_idx;
      music->idx = idx;
    }

    sec = idx / (2.0 * 44100.0);
    frame = sec * 29.97;

    if (frame != p_frame) {
      analyse_audio_frame(anal, music, idx, &audio_mean, &audio_peaks) ;

      //next_scene_num = markovStep(source->scene_motion_distance, current_scene_num, 1.0, 4.0);
      next_scene_num = markovStep(source->scene_palette_distance, current_scene_num, -1.0, 40.0);

      is_beat = false;

      if (audio_peaks > 1.35* p_audio_peaks) {
        is_beat = add_beat_to_rhythm(&beat_idx, idx, &beat_idx_delta) ;
        if (is_beat) last_beat_idx = idx;
      }
      p_audio_peaks = audio_peaks;

      if (idx - last_beat_idx > beat_idx_delta) {
        is_beat = true;
        last_beat_idx = idx;
      }

      if (is_beat) {
        current_scene = &source->scenes->at(next_scene_num);
        current_scene_num = next_scene_num;
        scene_start_frame = frame;
      }

      Mat footage;

      current_scene->current_frame_num++;
      if (current_scene->current_frame_num >= current_scene->start_frame_num + current_scene->duration-5) {
        current_scene->current_frame_num = current_scene->start_frame_num + current_scene->duration-5;
        current_scene->current_frame_num = current_scene->start_frame_num;
        printf ("****** REWIND *******\n");
      }

      if (derez) {
         source->derez_cap->set(CAP_PROP_POS_FRAMES, current_scene->current_frame_num);
        source->derez_cap->read(footage); 
      } else {
        source->raw_cap->set(CAP_PROP_POS_FRAMES, current_scene->current_frame_num);
        source->raw_cap->read(footage); 
      }

      footage.copyTo(screen);

      mask *= 0;
//      title(mask, source);


      multiply(screen, 0.5 + anal/512.0, screen);

      multiply(screen, mask/256.0, screen);
      blur(screen, screen, Size(2,2));
      screen += mask * 0.5 + footage*0.8;

      drawPaletteBox(current_scene, screen, 10,10, 50, true);
      drawPaletteBox(&source->scenes->at(next_scene_num), screen, 10,80, 50, true);


      showFrame(screen);

      printf ("%8.4fs ", sec);
      printf ("%6.2ffps ", 1.0 / (sec - p_sec));
      printf ("%8d ", music->idx);
      printf ("%06d ", frame);
      printf ("   %6.4f ", audio_mean);
      bars((audio_mean-0.2) * 2.0);
      printf (" %6.4f ", audio_peaks);
      bars(audio_peaks * 10.0);
      printf("%s ", is_beat ? " +++ " : "     ");
      printf ("%03d ", current_scene_num);
      printf ("\n");

      p_sec = sec;
      p_frame = frame;
    }

  }
}




int main( int argc, char** argv )
{
  ft2 = cv::freetype::createFreeType2();
  ft2->loadFontData( "futura1.ttf", 0 );

  play("diamondbay.clips.mov", "05 sense_5days later.ogg", false, true);

  return 0;
}
