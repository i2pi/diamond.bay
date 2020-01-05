#ifndef __CACHE_H__
#define __CACHE_H__

#include "diamond.h"

void openCache(sourceT *s, const char *filename="cache.json");
void saveCache(sourceT *s, const char *filename="cache.json");
void cacheScenes(sourceT *s);
void loadScenesFromCache(sourceT *s);

#endif
