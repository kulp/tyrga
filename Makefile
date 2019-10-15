# ensure Java 8 support, which implies StackMapTable
JAVAFLAGS += -target 1.8 -source 1.8
# avoid generating unused debugging information
JAVAFLAGS += -g:none

TAS ?= tas
TLD ?= tld

TYRGA_CLI := $(shell cargo with echo -- run --quiet --bin tyrga-cli --offline)

JAVA_SRC_DIRS += test test/interesting/module/name
JAVA_SRC_DIRS += tyrga-lib/java-lib/tyrga

# javac wants to put directory prefixes on output files even when we provide a
# `-d <outdir>` option, to reflect package hierarchy. Since it is difficult to
# determine ahead of time what that location will be, we just let javac put the
# generated class files next to the source files, as is its default behavior.
vpath %.java  $(JAVA_SRC_DIRS)
vpath %.class $(JAVA_SRC_DIRS)

ALL_JAVA = $(foreach d,$(JAVA_SRC_DIRS),$(wildcard $d/*.java))

classes: $(ALL_JAVA:%.java=%.class)

tases: $(ALL_JAVA:%.java=%.tas)

%.class: %.java
	javac $(JAVAFLAGS) $<

# Rebuild tases when the translator binary changes
%.tas: %.class $(TYRGA_CLI)
	$(TYRGA_CLI) translate --output $@ $<

vpath %.tas tyrga-lib/tenyr-lib

BUILTIN_java = $(wildcard tyrga-lib/java-lib/tyrga/*.java)
BUILTIN_tas = $(wildcard tyrga-lib/tenyr-lib/*.tas)

LIB_to += $(BUILTIN_tas:.tas=.to)
LIB_to += $(BUILTIN_java:.java=.to)

%.texe: %.to $(LIB_to)
	$(TLD) -o $@ $^

%.to: %.tas
	$(TAS) -o $@ $<

