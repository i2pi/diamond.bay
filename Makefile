LIBS=-lopencv_videoio -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_freetype -lopencv_video -lopencv_photo -lportaudio -lvorbis -lvorbisfile -logg
CPPFLAGS=-Wall -g -std=c++11
SRC=main.cpp audio.cpp

main: $(SRC) Makefile
	g++ $(CPPFLAGS) $(SRC) $(LIBS) -o main

bug: bug.cpp audio.cpp Makefile
	g++ $(CPPFLAGS) bug.cpp audio.cpp $(LIBS) -o bug

clean:
	rm -f main bug
