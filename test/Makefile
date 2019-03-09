SOURCES = $(wildcard *.java)
CLASSES = $(SOURCES:.java=.class)

JAVAC_FLAGS = -g:none

all: $(CLASSES)

%.class: %.java
	javac $(JAVAC_FLAGS) $<

clean:
	$(RM) $(CLASSES)
