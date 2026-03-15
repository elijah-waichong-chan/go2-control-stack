from arm_controller.d1_ik import DEFAULT_D1_URDF_NAME, resolve_d1_urdf_path


def test_resolve_d1_urdf_path_finds_repo_urdf():
    urdf_path = resolve_d1_urdf_path()
    assert urdf_path.is_file()
    assert urdf_path.name == DEFAULT_D1_URDF_NAME
