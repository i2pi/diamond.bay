LIBS=-lopencv_videoio -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_freetype -lopencv_video -lopencv_photo -lportaudio -lvorbis -lvorbisfile -logg
CPPFLAGS=-Wall -O3 -std=c++11
SRC=main.cpp audio.c

main: main.cpp Makefile
	g++ $(CPPFLAGS) $(SRC) $(LIBS) -o main

clean:
	rm -f main
		
segments: segments.cpp
	g++ -std=c++11 segments.cpp -lopencv_video -lopencv_videoio -lopencv_highgui -lopencv_core -lopencv_imgproc -o segments

kmeans: kmeans.cpp
	g++ -std=c++11 kmeans.cpp -lopencv_video -lopencv_videoio -lopencv_highgui -lopencv_core -lopencv_imgproc -o kmeans

lkflow: lkflow.cpp
	g++ -std=c++11 lkflow.cpp $(LIBS) -o lkflow

