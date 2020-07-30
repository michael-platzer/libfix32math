
CC = patmos-clang
AR = patmos-ar

LIBFIX32 = libfix32math.a
OBJ      = src/fix32math.o

CFLAGS ?= -target patmos-unknown-unknown-elf -O2 -I.

$(LIBFIX32): $(OBJ)
	$(AR) rcs $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $^

clean:
	rm -f $(LIBFIX32) $(OBJ)
