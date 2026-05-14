#include <kd3/kd3-c.h>

#include <stdio.h>

int main(){
/* Example with three points – expand the list for as many as you need */
    constexpr kd3_point_t points[] = {
        /* point 0 */ { .coords = {0.0f, 0.0f, 0.0f}, .payload_id = 0 },
        /* point 1 */ { .coords = {1.0f, 2.5f, -3.0f}, .payload_id = 1 },
        /* point 2 */ { .coords = {4.2f, -1.0f, 0.0f}, .payload_id = 2 }
        /* … add further entries … */
    };

    kd3_error_t error=0;
    auto tree = kd3_tree_create(points, sizeof(points)/sizeof(kd3_point_t), &error);
    if(error==0){
        kd3_knn_result_t storage[3];
        const float point[3] = {100,2,0};
        size_t items = 3;
        kd3_error_t found = kd3_tree_query_knn(tree, point, storage, &items);
        //Ignore the possible error for now...
        for(int i=0;i<items;i++){
            auto point = points[storage[i].payload_id].coords;
            printf("(%f,%f,%f)\n", point[0],point[1],point[2]);
        }

        kd3_tree_destroy(tree);
    }

    return 0;
}