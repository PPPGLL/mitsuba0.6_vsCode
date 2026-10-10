#include <H5Cpp.h>
#include <vector>
#include <array>
#include <cstdint>
#include <string>
#include <iostream>

class HDF5Writer {
public:
	HDF5Writer(const std::string& filename, bool overwrite = true)
		: file(filename, overwrite ? H5F_ACC_TRUNC : H5F_ACC_EXCL) {}

	void createNewFile(const std::string& filename, bool overwrite = true) {
		file.close();
		file = H5::H5File(filename, overwrite ? H5F_ACC_TRUNC : H5F_ACC_EXCL);
	}

	void writeFloatDataset(const std::string& datasetName, const std::vector<float>& data) {
		hsize_t dims[1] = { data.size() };
		H5::DataSpace dataspace(1, dims);
		H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_FLOAT, dataspace);
		dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);
	}

	void writeUInt32Dataset(const std::string& datasetName, const std::vector<uint32_t>& data) {
		hsize_t dims[1] = { data.size() };
		H5::DataSpace dataspace(1, dims);
		H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_UINT, dataspace);
		dataset.write(data.data(), H5::PredType::NATIVE_UINT);
	}

	void writeInt32Dataset(const std::string& datasetName, const std::vector<int32_t>& data) {
		hsize_t dims[1] = { data.size() };
		H5::DataSpace dataspace(1, dims);
		H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_INT, dataspace);
		dataset.write(data.data(), H5::PredType::NATIVE_INT);
	}

	void writeFloatArray(const std::string& datasetName, const std::vector<std::vector<float>>& data) {
		hsize_t dims[2] = { data.size(), data[0].size() };
		H5::DataSpace dataspace(2, dims);
		H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_FLOAT, dataspace);

		std::vector<float> flatData;
		for (const auto& row : data) {
			flatData.insert(flatData.end(), row.begin(), row.end());
		}

		dataset.write(flatData.data(), H5::PredType::NATIVE_FLOAT);
	}

	void writeFloatArray(const std::string& datasetName, const float* data, size_t size) {
		hsize_t dims[1] = { size };
		H5::DataSpace dataspace(1, dims);
		H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_FLOAT, dataspace);
		dataset.write(data, H5::PredType::NATIVE_FLOAT);
	}

	void writeVector3Array(const std::string& datasetName, const std::vector<std::array<float, 3>>& vectors) {
		hsize_t dims[2] = { vectors.size(), 3 };
		H5::DataSpace dataspace(2, dims);
		H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_FLOAT, dataspace);

		std::vector<float> flatData;
		for (const auto& vec : vectors) {
			flatData.insert(flatData.end(), vec.begin(), vec.end());
		}

		dataset.write(flatData.data(), H5::PredType::NATIVE_FLOAT);
	}

	void writeVector2Array(const std::string& datasetName, const std::vector<std::array<float, 2>>& vectors) {
		hsize_t dims[2] = { vectors.size(), 2 };
		H5::DataSpace dataspace(2, dims);
		H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_FLOAT, dataspace);

		std::vector<float> flatData;
		for (const auto& vec : vectors) {
			flatData.insert(flatData.end(), vec.begin(), vec.end());
		}

		dataset.write(flatData.data(), H5::PredType::NATIVE_FLOAT);
	}


private:
	H5::H5File file;
};

