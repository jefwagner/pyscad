/// @file utils.h 
//
// Copyright (c) 2023, Jef Wagner <jefwagner@gmail.com>
//
// This file is part of pyscad.
//
// pyscad is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//
// pyscad is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// pyscad. If not, see <https://www.gnu.org/licenses/>.
//

#ifndef PYSCAD_UTILS_H
#define PYSCAD_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#import <stddef.h>

typedef struct _DynArray {
    size_t size;
    size_t num;
    void* data;
} DynArray;

void* new_dyn_array(size_t element_size,)

#ifdef __cplusplus
}
#endif

#endif //PYSCAD_UTILS_H
