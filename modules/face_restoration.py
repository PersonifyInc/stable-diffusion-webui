from modules import shared


class FaceRestoration:
    def name(self):
        return "None"

    def restore(self, np_image):
        return np_image

    def release(self):
        pass

def restore_faces(np_image):
    face_restorers = [x for x in shared.face_restorers if x.name() == shared.opts.face_restoration_model or shared.opts.face_restoration_model is None]
    if len(face_restorers) == 0:
        return np_image

    face_restorer = face_restorers[0]

    return face_restorer.restore(np_image)

def release_model():
    face_restorers = [x for x in shared.face_restorers if x.name() == shared.opts.face_restoration_model or shared.opts.face_restoration_model is None]
    if len(face_restorers) == 0:
        return

    for face_restorer in face_restorers:
        face_restorer.release_model()