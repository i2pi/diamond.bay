#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "portaudio.h"
#include "vorbis/codec.h"
#include "vorbis/vorbisfile.h"

#include "audio.h"

#ifndef DIST
void pa_error(PaError err) 
{
	if (err == paNoError) return;

	Pa_Terminate();
	fprintf( stderr, "An error occured while using the portaudio stream\n" );
	fprintf( stderr, "Error number: %d\n", err );
	fprintf( stderr, "Error message: %s\n", Pa_GetErrorText( err ) );
	exit (-1);
}
#endif

static int patestCallback (const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *userData) 
{
	int16_t *out = (int16_t *) outputBuffer;
	audioFileT *audio = (audioFileT *) userData;
	unsigned int i;

	for (i=0; i<framesPerBuffer; i++) {
		*out++ = audio->sample[audio->idx++]; //left
		if (audio->idx >= audio->samples) audio->idx = 0;
        *out++ = audio->sample[audio->idx++];
		if (audio->idx >= audio->samples) audio->idx = 0;
	}

	return (0);
}

audioFileT *ogg_to_af (OggVorbis_File *vf) 
{
	int current_section;
	vorbis_info *vi;
	audioFileT	*audio;
	static char	*data = NULL;
	long	length;
	long	ret;
	int		eof;

	if (!data) {
		data = (char *) malloc (MAX_AUDIO_LENGTH);
		if (!data) {
#ifndef DIST
			fprintf (stderr, "Couldn't create ogg pcm buffer\n");
#endif
			exit (-1);
		}
	}

	// char **ptr = ov_comment(&vf,-1)->user_comments;
	vi = ov_info(vf,-1);

	audio = (audioFileT *) malloc (sizeof(audioFileT));
	if (!audio) {
#ifndef DIST
		fprintf (stderr,"Couldn't malloc audio file\n");
#endif
		exit (-1);
	} 

	eof = 0;
	length = 0;
	while (!eof) {
		ret = ov_read(vf, &data[length],4096,0,2,1,&current_section);
		if (ret == 0) {
			eof = 1;
		} else 
		if (ret < 0) {
#ifndef DIST
			fprintf (stderr, "Some shits broke\n");
#endif
			exit (-1);
		} else {
			length += ret;
		}	
		if (length >= MAX_AUDIO_LENGTH - 4097) { 
			eof = 1;
#ifndef DIST
			printf ("Truncating ogg -- too long\n");
#endif
		}
	}

	audio->idx = 0;
	audio->samples = length / sizeof(int16_t);
	audio->sample = (int16_t *) malloc (sizeof(int16_t) * audio->samples);
	memcpy (audio->sample, data, length);

	ov_clear(vf);

	return (audio);	
}

audioFileT *read_ogg(const char *fname) 
{
	OggVorbis_File vf;
	FILE	*fp;
	audioFileT	*audio;

	fp = fopen (fname, "r");
	if (!fp) {
#ifndef DIST
		fprintf (stderr, "Can't open input ogg file [%s]\n", fname);
#endif
		exit (-1);
	}

	if(ov_open_callbacks(fp, &vf, NULL, 0, OV_CALLBACKS_NOCLOSE) < 0) {
#ifndef DIST
		fprintf(stderr,"Input does not appear to be an Ogg bitstream.\n");
#endif
		exit(1);
	}

	audio = ogg_to_af(&vf);	

	return (audio);
}

typedef struct {
	unsigned char	*data;
	size_t	len;
	size_t	idx;
} data_wrapperT;


size_t read_func (void *ptr, size_t size, size_t nmemb, void *datasource) 
{	
	data_wrapperT *wrap = (data_wrapperT *) datasource;
	size_t	n = size * nmemb;
	size_t	sz = n;

	if (wrap->idx + n >= wrap->len) {
		sz = wrap->len - wrap->idx;
	}

	memcpy (ptr, &wrap->data[wrap->idx], sz);
	wrap->idx += sz;

	return (sz);
}


audioFileT *read_ogg_data(unsigned char *data, size_t len) 
{
	OggVorbis_File vf;
	ov_callbacks	cb;

	data_wrapperT 	wrap;
	audioFileT	*audio;

	wrap.data = data;
	wrap.idx = 0;
	wrap.len = len;

	cb.read_func = read_func;
	cb.seek_func = NULL;
	cb.tell_func = NULL;
	cb.close_func = NULL;

	if(ov_open_callbacks(&wrap, &vf, NULL, 0, cb) < 0) {
#ifndef DIST
		fprintf(stderr,"Input does not appear to be an Ogg bitstream.\n");
#endif
		exit(1);
	}

	audio = ogg_to_af(&vf);	

	return (audio);
}



PaStream *init_portaudio (int channels, int rate, void *data)
{
	PaStream *stream;
	PaError pa_err;

#ifndef DIST
	pa_error(Pa_Initialize());
#endif
	pa_err = Pa_OpenDefaultStream (&stream, 
			0,  // no input channels
			2, // stereo out
			paInt16, 
			44100,
			256,	// frames per buffer
			patestCallback,
			data);
#ifndef DIST
	pa_error(pa_err);
	pa_error(Pa_StartStream(stream));
#endif

	return (stream);
}

void close_portaudio (PaStream *stream) 
{
#ifndef DIST
	pa_error(Pa_StopStream(stream));
	pa_error(Pa_CloseStream(stream));
#else
	Pa_StopStream(stream);
	Pa_CloseStream(stream);
#endif
	Pa_Terminate();
}

