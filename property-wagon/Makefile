.DEFAULT_GOAL := default

run_getdata:
	python -c 'from ml_logic.data import get_data_with_cache; get_data_with_cache()'

run_preprocess:
	python -c 'from interface.main import preprocess; preprocess()'

test_gcp_setup:
	@pytest \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_project \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_project_id
