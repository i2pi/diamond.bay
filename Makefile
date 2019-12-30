LIBS=-lopencv_videoio -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_freetype -lopencv_video -lopencv_photo -lportaudio -lvorbis -lvorbisfile -logg
CPPFLAGS=-Wall -g -std=c++11
SRC=main.cpp audio.cpp
PROG=diamond.bay

$(PROG): $(SRC) Makefile
	g++ $(CPPFLAGS) $(SRC) $(LIBS) -o $(PROG)

clean:
	rm -f $(PROG)
