/*
 * Copyright (c) 2016, Peter Fankhauser, Christian Gehring, Hannes Sommer, Paul Furgale, Remo Diethelm
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Autonomous Systems Lab, ETH Zurich nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Peter Fankhauser, Christian Gehring, Hannes Sommer, Paul Furgale,
 * Remo Diethelm BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
*/
/*
 * tf_kindr_test.cpp
 *
 *  Created on: Jun 17, 2016
 *      Author: Christian Gehring
 */

#include <gtest/gtest.h>
#include <kindr/Core>
#include <kindr/rotations/gtest_rotations.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace kindr
{

template<>
class RotationConversion<tf2::Quaternion, tf2::Vector3, double>
{
  typedef tf2::Quaternion Rotation;
  typedef tf2::Vector3 Vector;

public:
  inline static void convertToOtherRotation(
    Rotation & out,
    const kindr::RotationQuaternion<double> & in)
  {
    kindr::RotationQuaternion<double> in2 = in;
    out = tf2::Quaternion(in2.x(), in2.y(), in2.z(), in2.w());
  }

  inline static void convertToKindr(kindr::RotationQuaternion<double> & out, Rotation & in)
  {
    out = kindr::RotationQuaternion<double>(in.w(), in.x(), in.y(), in.z());
  }

  inline static void convertToVelocityVector(Vector & out, const Eigen::Matrix<double, 3, 1> & in)
  {
    out = tf2::Vector3(in.x(), in.y(), in.z());
  }

  inline static void concatenate(Rotation & res, const Rotation & rot1, const Rotation & rot2)
  {
    res = rot2 * rot1;
  }

  inline static void getRotationMatrixFromRotation(
    Eigen::Matrix3d & rotationMatrix,
    const Rotation & quaternion)
  {
    tf2::Matrix3x3 tfRotationmatrix;
    tfRotationmatrix.setRotation(quaternion);
    for (int i = 0; i < 3; i++) {
      rotationMatrix(i, 0) = tfRotationmatrix.getRow(i).x();
      rotationMatrix(i, 1) = tfRotationmatrix.getRow(i).y();
      rotationMatrix(i, 2) = tfRotationmatrix.getRow(i).z();
    }
  }

  inline static void rotateVector(
    Eigen::Matrix<double, 3, 1> & A_r, const Rotation & rotationBToA,
    const Eigen::Matrix<double, 3, 1> & B_r)
  {
    tf2::Vector3 A_v = tf2::quatRotate(rotationBToA, tf2::Vector3(B_r.x(), B_r.y(), B_r.z()));
    A_r.x() = A_v.x();
    A_r.y() = A_v.y();
    A_r.z() = A_v.z();
  }
};


template<>
class RotationConversion<tf2::Matrix3x3, tf2::Vector3, double>
{
  typedef tf2::Matrix3x3 Rotation;
  typedef tf2::Vector3 Vector;

public:
  inline static void convertToOtherRotation(
    Rotation & out,
    const kindr::RotationQuaternion<double> & in)
  {
    kindr::RotationQuaternion<double> in2 = in;
    out.setRotation(tf2::Quaternion(in2.x(), in2.y(), in2.z(), in2.w()));
  }

  inline static void convertToKindr(kindr::RotationQuaternion<double> & out, Rotation & matrix)
  {
    tf2::Quaternion quat;
    matrix.getRotation(quat);
    out = kindr::RotationQuaternion<double>(quat.w(), quat.x(), quat.y(), quat.z());
  }

  inline static void convertToOtherVelocityVector(
    Vector & out,
    const Eigen::Matrix<double, 3, 1> & in)
  {
    out = tf2::Vector3(in.x(), in.y(), in.z());
  }

  inline static void concatenate(Rotation & res, const Rotation & rot1, const Rotation & rot2)
  {
    res = rot2 * rot1;
  }

  inline static void getRotationMatrixFromRotation(
    Eigen::Matrix3d & rotationMatrix,
    const Rotation & matrix)
  {
    for (int i = 0; i < 3; i++) {
      rotationMatrix(i, 0) = matrix.getRow(i).x();
      rotationMatrix(i, 1) = matrix.getRow(i).y();
      rotationMatrix(i, 2) = matrix.getRow(i).z();
    }
  }

  inline static void rotateVector(
    Eigen::Matrix<double, 3, 1> & A_r, const Rotation & rotationBToA,
    const Eigen::Matrix<double, 3, 1> & B_r)
  {
    tf2::Vector3 B_v(B_r.x(), B_r.y(), B_r.z());
    tf2::Vector3 A_v = rotationBToA * B_v;
    A_r.x() = A_v.x();
    A_r.y() = A_v.y();
    A_r.z() = A_v.z();
  }
};


}  // namespace kindr
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TfConventionTest, Concatenation) {
  kindr::ConventionTest<tf2::Quaternion, tf2::Vector3, double>::testConcatenation();
  kindr::ConventionTest<tf2::Matrix3x3, tf2::Vector3, double>::testConcatenation();
}

TEST(TfConventionTest, Rotation) {
  kindr::ConventionTest<tf2::Quaternion, tf2::Vector3, double>::testRotationMatrix();
  kindr::ConventionTest<tf2::Matrix3x3, tf2::Vector3, double>::testRotationMatrix();
}

// TEST(TfConventionTest, BoxPlus) {
//  kindr::ConventionTest<tf2::Quaternion, tf2::Vector3, double>::testBoxPlus();
// }

TEST(TfConventionTest, GeometricalInterpretation) {
  kindr::ConventionTest<tf2::Quaternion, tf2::Vector3, double>::testGeometricalInterpretation();
  kindr::ConventionTest<tf2::Matrix3x3, tf2::Vector3, double>::testGeometricalInterpretation();
}
