`kd3` ships its C interface as a C++ stb-style library.  
You are either link the default implementation which is automatically built as part of this xmake project, or quickly integrate it within your own.  
[kd3-c.cpp](../lib/kd3/kd3-c.cpp) offers an example of how this is done:

```c++
#define KD3_CXX_IMPL
#include <kd3/kd3-c.h>
```

You just need to write this within a C++ file. Then, to use it just make sure to include `<kd3/kd3-c.h>` when needed.  
This basic example works just fine, but you might want to customize the library. You can have multiple instances with different parametrization within the same project if you want.

The easies way is to introduce a custom header (`.h`) for each variant:

```c
//Define here all macros. At the very least this is needed if you want more than one version.

#define KD3_NS kd3_f32

#include <kd3/kd3-c.h>
```

```c
//In this case we are setting up the base type to int8_t.
//Check the header file itself to see all the available flags and variables.

#define KD3_NS kd3_i8
#define KD3_BASE_TYPE int8_t

#include <kd3/kd3-c.h>
```

Then, in some C++ file of your sourcecode:

```c++
#define KD3_CXX_IMPL
#include "kd3_f32.h"

#define KD3_CXX_IMPL
#include "kd3_i8.h"
```

And you can just make use of the custom headers where needed.