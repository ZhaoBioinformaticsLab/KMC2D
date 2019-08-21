SOURCE = KMC2D.cu
OBJECT = KMC2D.o
TARGET = KMC2D
CC = nvcc#g++

CMP_OPT  = -O2 
LINK_OPT = 

$(TARGET):$(OBJECT)
	$(CC) $(LINK_OPT) -o $(TARGET) $(OBJECT)

$(OBJECT):$(SOURCE)
	$(CC) $(CMP_OPT) -c $(SOURCE)

clean:
	rm -f $(TARGET) $(OBJECT) 
