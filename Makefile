# ensure Java 8 support, which implies StackMapTable
JAVACFLAGS += --release 11
# avoid generating unused debugging information
JAVACFLAGS += -g:none

JAVAC ?= javac
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

ALL_JAVA = $(foreach d,$(JAVA_SRC_DIRS),$(filter-out %Test.java,$(wildcard $d/*.java)))

classes: $(ALL_JAVA:%.java=%.class)

tases: $(ALL_JAVA:%.java=%.tas)

%.class: %.java
	$(JAVAC) $(JAVACFLAGS) $<

# Rebuild tases when the translator binary changes
%.tas: %.class $(TYRGA_CLI)
	$(TYRGA_CLI) translate --output $@ $<

vpath %.tas tyrga-lib/tenyr-lib

BUILTIN_java = $(wildcard tyrga-lib/java-lib/tyrga/*.java)
BUILTIN_tas = $(wildcard tyrga-lib/tenyr-lib/*.tas)

# Entry point comes first
LIB_to  = $(filter %/Entry.to,$(BUILTIN_tas:.tas=.to))
LIB_to += $(filter-out %/Entry.to,$(BUILTIN_tas:.tas=.to))
LIB_to += $(BUILTIN_java:.java=.to)

# This rule deletes the contents of the target-specific output directory before
# proceeding. Since it does not have the ability to know exactly which objects
# will be generated from a given .java file, it must use wildcards, and if it
# did not delete the contents first, files could creep into the $TARGET_DIR and
# cause hard-to-reproduce build states.
%.texe: TARGET_DIR = $*_files
%.texe: %.java $(LIB_to)
	$(RM) -r $(TARGET_DIR)
	mkdir -p $(TARGET_DIR)
	$(JAVAC) $(JAVACFLAGS) -d $(TARGET_DIR) $<
	basename -s .class -a $(TARGET_DIR)/*.class | xargs -I{} $(MAKE) $(TARGET_DIR)/{}.to
	$(TLD) -o $@ $(filter %.to,$^) $(TARGET_DIR)/*.to
	$(RM) -r $(TARGET_DIR)

%.to: %.tas
	$(TAS) -o $@ $<

test/Native.texe: test/Native_support.to

JUNIT_JAR_URL = https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.11.0/junit-platform-console-standalone-1.11.0.jar
JUNIT_JAR = test-lib/$(notdir $(JUNIT_JAR_URL))

$(JUNIT_JAR):
	mkdir -p $(@D)
	wget -O $@ $(JUNIT_JAR_URL)

.PHONY: check
check: check_java

.PHONY: check_java
check_java: check_Builtin

.PHONY: check_Builtin
check_Builtin: check_%: %.java %Test.java $(JUNIT_JAR)
	javac -d $@ --class-path $(JUNIT_JAR) $(filter %.java,$^)
	java -jar $(JUNIT_JAR) execute --class-path $@ --select-class tyrga.$*Test
