import json
import lookup_table


def get_bboxes(frames) -> list:
    all_bboxes = []
    for frame in frames:
        bboxes = []
        bbox = dict()
        for b in frame['annotations']:
            bbox['timestamp'] = frame['timestamp']
            bbox['x'] = b['position']['x']
            bbox['y'] = b['position']['y']
            bbox['z'] = b['position']['z']
            bbox['dx'] = b['dimensions']['x']
            bbox['dy'] = b['dimensions']['y']
            bbox['dz'] = b['dimensions']['z']
            bbox['yaw'] =b['yaw']
            bbox['class_id'] = b['category_id']
            bboxes.append(bbox.copy())
        all_bboxes.append(bboxes)
    return all_bboxes

