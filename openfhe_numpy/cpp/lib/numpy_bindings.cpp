//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2025, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================
#include <pybind11/pybind11.h>
#include "numpy_enc_matrix.h"

using namespace openfhe_numpy;
using namespace lbcrypto;


namespace py = pybind11;
void bind_enums_and_constants(py::module& m);
void bind_matrix_funcs(py::module& m);

//  [tango] Since the current versions of openfhe_python and OpenFHE don’t support this binding, I’ve included it here for testing purposes.
void bind_ciphertext(py::module& m);
void bind_privatekey(py::module& m);

PYBIND11_MODULE(openfhe_numpy, m) {
    m.doc() = "OpenFHE-Numpy C++ extension";

    // Version info comes from CMake-generated config.h
    // m.attr("__version__")     = OPENFHE_NUMPY_VERSION;
    m.attr("__author__")      = "OpenFHE-Numpy Team";
    m.attr("__description__") = "Python bindings for OpenFHE-Numpy homomorphic operations";
    m.attr("__license__")     = "MIT";

// Add OpenFHE version if available
#ifdef OPENFHE_VERSION
    // Different OpenFHE versions define the version differently
    #ifdef BASE_OPENFHE_VERSION
    m.attr("openfhe_version") = BASE_OPENFHE_VERSION;
    #else
    // Fallback to a string version or try direct access
    m.attr("openfhe_version") = "Compatible with OpenFHE 1.2.3+";
    #endif
#endif

    bind_enums_and_constants(m);
    bind_matrix_funcs(m);
    bind_ciphertext(m);
    bind_privatekey(m);
}

void bind_enums_and_constants(py::module& m) {
    // Linear Transformation Types
    py::enum_<LinTransType>(m, "LinTransType")
        .value("SIGMA", LinTransType::SIGMA)
        .value("TAU", LinTransType::TAU)
        .value("PHI", LinTransType::PHI)
        .value("PSI", LinTransType::PSI)
        .value("TRANSPOSE", LinTransType::TRANSPOSE)
        .export_values();
    py::implicitly_convertible<int, LinTransType>();

    // Matrix Vector Multiplication Types
    py::enum_<MatVecEncoding>(m, "MatVecEncoding")
        .value("MM_CRC", MatVecEncoding::MM_CRC)
        .value("MM_RCR", MatVecEncoding::MM_RCR)
        .value("MM_DIAG", MatVecEncoding::MM_DIAG)
        .export_values();
    py::implicitly_convertible<int, MatVecEncoding>();

    py::enum_<ArrayEncodingType>(m, "ArrayEncodingType")
        .value("ROW_MAJOR", ArrayEncodingType::ROW_MAJOR)
        .value("COL_MAJOR", ArrayEncodingType::COL_MAJOR)
        .value("DIAG_MAJOR", ArrayEncodingType::DIAG_MAJOR)
        .export_values();
    py::implicitly_convertible<int, ArrayEncodingType>();
}

void bind_matrix_funcs(py::module& m) {
    // MulDepthAccumulation
    m.def("MulDepthAccumulation",
        &MulDepthAccumulation,
        py::arg("numRows"),
        py::arg("numCols"),
        py::arg("isSumRows"),
        "Compute the CKKS multiplicative-depth needed to sum over a numRows x numCols matrix, optionally summing rows.");

    // EvalLinTransKeyGen
    m.def("EvalLinTransKeyGen",
        [](PrivateKey<DCRTPoly>& secretKey, int32_t numCols, LinTransType type, int32_t numRepeats) {
            EvalLinTransKeyGen(secretKey, numCols, type, numRepeats);
        },
        py::arg("secretKey"),
        py::arg("numCols"),
        py::arg("type"),
        py::arg("numRepeats") = 0);

    m.def("EvalSumCumRowsKeyGen",
        [](PrivateKey<DCRTPoly>& secretKey, int32_t numCols) {
            EvalSumCumRowsKeyGen(secretKey, numCols);
        },
        py::arg("secretKey"),
        py::arg("numCols"));

    m.def("EvalSumCumColsKeyGen",
        [](PrivateKey<DCRTPoly>& secretKey, int32_t numCols) {
            EvalSumCumColsKeyGen(secretKey, numCols);
        },
        py::arg("secretKey"),
        py::arg("numCols"));

    m.def("EvalLinTransSigma",
        static_cast<Ciphertext<DCRTPoly>(*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(&EvalLinTransSigma),
        py::arg("secretKey"),
        py::arg("ciphertext"),
        py::arg("numCols"));

    m.def("EvalLinTransSigma",
        static_cast<Ciphertext<DCRTPoly>(*)(const Ciphertext<DCRTPoly>&, int32_t)>(&EvalLinTransSigma),
        py::arg("ciphertext"),
        py::arg("numCols"));

    m.def("EvalLinTransTau",
        static_cast<Ciphertext<DCRTPoly>(*)(const Ciphertext<DCRTPoly>&, int32_t)>(&EvalLinTransTau),
        py::arg("ctVector"),
        py::arg("numCols"));

    m.def("EvalLinTransTau",
        static_cast<Ciphertext<DCRTPoly>(*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(&EvalLinTransTau),
        py::arg("secretKey"),
        py::arg("ciphertext"),
        py::arg("numCols"));


    m.def("EvalLinTransPhi",
        static_cast<Ciphertext<DCRTPoly>(*)(const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(&EvalLinTransPhi),
        py::arg("ctVector"),
        py::arg("numCols"),
        py::arg("numRepeats"));

    m.def("EvalLinTransPhi",
        static_cast<Ciphertext<DCRTPoly>(*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(&EvalLinTransPhi),
        py::arg("secretKey"),
        py::arg("ctVector"),
        py::arg("numCols"),
        py::arg("numRepeats"));

    m.def("EvalLinTransPsi",
        static_cast<Ciphertext<DCRTPoly>(*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(&EvalLinTransPsi),
        py::arg("secretKey"),
        py::arg("ctVector"),
        py::arg("numCols"),
        py::arg("numRepeats"));

    m.def("EvalLinTransPsi",
        static_cast<Ciphertext<DCRTPoly>(*)(const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(&EvalLinTransPsi),
        py::arg("ctVector"),
        py::arg("numCols"),
        py::arg("numRepeats"));

    m.def("EvalMatMulSquare",
        [](const Ciphertext<DCRTPoly>& matrix_a, const Ciphertext<DCRTPoly>& matrixB, int32_t numCols) {
            return EvalMatMulSquare(matrix_a, matrixB, numCols);
        },
        py::arg("matrix_a"),
        py::arg("matrixB"),
        py::arg("numCols"));

    m.def("EvalTranspose",
        static_cast<Ciphertext<DCRTPoly>(*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(&EvalTranspose),
        py::arg("secretKey"),
        py::arg("ciphertext"),
        py::arg("numCols"));

    m.def("EvalTranspose",
        static_cast<Ciphertext<DCRTPoly>(*)(const Ciphertext<DCRTPoly>&, int32_t)>(&EvalTranspose),
        py::arg("ciphertext"),
        py::arg("numCols"));

    m.def("EvalSquareMatMultRotateKeyGen",
        [](PrivateKey<DCRTPoly>& secretKey, int32_t numCols) {
            EvalSquareMatMultRotateKeyGen(secretKey, numCols);
        },
        py::arg("secretKey"),
        py::arg("numCols"));

    m.def("EvalMultMatVec",
        [](std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<DCRTPoly>>>& evalKeys, MatVecEncoding encodeType,
        int32_t numCols, const Ciphertext<DCRTPoly>& ctVector, const Ciphertext<DCRTPoly>& ctMatrix) {
            return EvalMultMatVec(evalKeys, encodeType, numCols, ctVector, ctMatrix);
        },
        py::arg("evalKeys"),
        py::arg("encodeType"),
        py::arg("numCols"),
        py::arg("ctVector"),
        py::arg("ctMatrix"));

    m.def("EvalSumCumRows",
        [](const Ciphertext<DCRTPoly>& ciphertext, uint32_t numCols, uint32_t numRows, uint32_t slots) {
            return EvalSumCumRows(ciphertext, numCols, numRows, slots);
        },
        py::arg("ciphertext"),
        py::arg("numCols"),
        py::arg("numRows") = 0,
        py::arg("slots") = 0);

    m.def("EvalSumCumCols",
        [](const Ciphertext<DCRTPoly>& ciphertext, uint32_t numCols, uint32_t subringDim) {
            return EvalSumCumCols(ciphertext, numCols, subringDim);
        },
        py::arg("ciphertext"),
        py::arg("numCols"),
        py::arg("subringDim") = 0);

    m.def("EvalReduceCumRows",
        [](const Ciphertext<DCRTPoly>& ciphertext, uint32_t numCols, uint32_t numRows, uint32_t slots) {
            return EvalReduceCumRows(ciphertext, numCols, numRows, slots);
        },
        py::arg("ciphertext"),
        py::arg("numCols"),
        py::arg("numRows") = 0,
        py::arg("slots")   = 0);

    m.def("EvalReduceCumCols",
        [](const Ciphertext<DCRTPoly>& ciphertext, uint32_t numCols, uint32_t subringDim) {
            return EvalReduceCumCols(ciphertext, numCols, subringDim);
        },
        py::arg("ciphertext"),
        py::arg("numCols"),
        py::arg("subringDim") = 0);
}

//  [tango] Since the current versions of openfhe_python and OpenFHE don’t support this binding, I’ve included it here for testing purposes.
void bind_ciphertext(py::module& m) {
    py::object existingModule = py::module_::import("openfhe");
    py::object pyClsObj       = existingModule.attr("Ciphertext");
    auto cls                  = py::reinterpret_borrow<py::class_<Ciphertext<DCRTPoly>>>(pyClsObj);
    cls.def("GetEncodingType", [](const Ciphertext<DCRTPoly>& ct) { return ct->GetEncodingType(); });
    cls.def("GetCryptoContext", [](const Ciphertext<DCRTPoly>& ct) { return ct->GetCryptoContext(); });
}

void bind_privatekey(py::module& m) {
    py::object existingModule = py::module_::import("openfhe");
    py::object pyClsObj       = existingModule.attr("PrivateKey");
    auto cls                  = py::reinterpret_borrow<py::class_<PrivateKey<DCRTPoly>>>(pyClsObj);
    cls.def("GetCryptoContext", [](const PrivateKey<DCRTPoly>& ct) { return ct->GetCryptoContext(); });
}


// FOR LATER
// void bind_metadata(py::module& m) {
//     py::class_<ArrayMetadata, Metadata, std::shared_ptr<ArrayMetadata>>(m, "ArrayMetadata")

//         .def(py::init<std::array<int,2>, int32_t, int32_t, int32_t, ArrayEncodingType>(),
//              py::arg("initialShape"),
//              py::arg("ndim"),
//              py::arg("rowsize"),
//              py::arg("batchSize"),
//              py::arg("encodeType") = ArrayEncodingType::ROW_MAJOR)

//         // properties for all fields
//         // properties, using lambdas to disambiguate
//         .def_property(
//             "initialShape",
//             [](const ArrayMetadata& a) -> std::array<int,2> { return a.initialShape(); },
//             [](ArrayMetadata& a, std::array<int,2> v) { a.initialShape() = v; },
//             "Original (flattened) array length")
//         .def_property(
//             "ndim",
//             [](const ArrayMetadata& a) -> int32_t { return a.ndim(); },
//             [](ArrayMetadata& a, int32_t v) { a.ndim() = v; },
//             "Number of dimensions")
//         .def_property(
//             "rowsize",
//             [](const ArrayMetadata& a) -> int32_t { return a.rowsize(); },
//             [](ArrayMetadata& a, int32_t v) { a.rowsize() = v; },
//             "Number of columns")
//         .def_property(
//             "nrows",
//             [](const ArrayMetadata& a) -> int32_t { return a.nrows(); },
//             [](ArrayMetadata& a, int32_t v) { a.nrows() = v; },
//             "Number of rows")
//         .def_property(
//             "batchSize",
//             [](const ArrayMetadata& a) -> int32_t { return a.batchSize(); },
//             [](ArrayMetadata& a, int32_t v) { a.batchSize() = v; },
//             "Batch size")
//         .def_property(
//             "encodeType",
//             [](const ArrayMetadata& a) -> ArrayEncodingType { return a.encodeType(); },
//             [](ArrayMetadata& a, ArrayEncodingType e) { a.encodeType() = e; },
//             "Row‐ or column‐major encoding")

//         // Metadata interface
//         .def("Clone", &ArrayMetadata::Clone, "Return a deep copy of this ArrayMetadata")
//         .def(
//             "__eq__",
//             [](const ArrayMetadata& a, const Metadata& b) { return a == b; },
//             py::is_operator(),
//             "Compare metadata for equality")
//         .def(
//             "print",
//             [](const ArrayMetadata& a) {
//                 std::ostringstream os;
//                 a.print(os);
//                 return os.str();
//             },
//             "Stringify the metadata")

//         // Serialization
//         .def("SerializedObjectName", &ArrayMetadata::SerializedObjectName)
//         .def_static("SerializedVersion", &ArrayMetadata::SerializedVersion);

//     // 3) If you need the template helpers for DCRTPoly:
//     m.def("GetArrayMetadata_DCRTPoly",
//           &ArrayMetadata::GetMetadata<DCRTPoly>,
//           py::arg("ciphertext"),
//           "Retrieve ArrayMetadata from a Ciphertext<DCRTPoly>");
//     m.def("StoreArrayMetadata_DCRTPoly",
//           &ArrayMetadata::StoreMetadata<DCRTPoly>,
//           py::arg("ciphertext"),
//           py::arg("metadata"),
//           "Attach ArrayMetadata to a Ciphertext<DCRTPoly>");
// }
