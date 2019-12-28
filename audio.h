#ifndef __AUDIO_H__
#define __AUDIO_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "portaudio.h"
#include "vorbis/codec.h"
#include "vorbis/vorbisfile.h"

#define MAX_AUDIO_LENGTH	(44100*4*10*60)

typedef struct {
	int16_t *sample;
	unsigned long samples;	
	int		idx;
} audioFileT;

audioFileT *read_ogg(const char *fname);
audioFileT *read_ogg_data(unsigned char *data, size_t len);

PaStream *init_portaudio (int channels, int rate, void *data);
void close_portaudio (PaStream *stream);

#endif
