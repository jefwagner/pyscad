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
#import <stdlib.h>

#define INITIAL_VEC_SIZE

// Generalized dynamic sized vector

typedef struct _Vec {
    size_t data_size;
    size_t size;
    size_t num;
    void* data;
} Vec;

inline void vec_new(Vec* vec, size_t data_size) {
    vec->data_size = data_size;
    vec->size = INITIAL_VEC_SIZE;
    vec->num = 0;
    vec->data = malloc(vec->size*data_size);
}

inline int vec_push(Vec* vec, void* elem) {
    void* new_data;
    if (vec->size == vec->num+1) {
        new_data = realloc(vec->data, 2*vec->size*vec->data_size);
        if (new_data == NULL) {
            return 1;
        }
        vec->size *= 2;
        vec->data = new_data;
    }
    memcpy(&(vec->data[vec->num]), elem, vec->data_size);
    vec->num += 1;
    return 0;
}

inline int vec_pop(vec* vec, void* elem) {
    memcpy(elem, &(vec->data[vec->num]), vec->data_size);
}

inline void vec_free(Vec* vec) {
    free(self->data);
}

#ifdef __cplusplus
}
#endif

#endif //PYSCAD_UTILS_H
