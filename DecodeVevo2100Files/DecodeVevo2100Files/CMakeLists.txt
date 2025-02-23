###############################################################################
#  Copyright Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

cmake_minimum_required(VERSION 2.8)

set( MODULE_NAME DecodeVevo2100Files )

project( ${MODULE_NAME} )

find_package( ITK REQUIRED )
include( ${ITK_USE_FILE} )

find_package( SlicerExecutionModel REQUIRED )
include( ${SlicerExecutionModel_USE_FILE} )

SEMMacroBuildCLI(
  NAME ${MODULE_NAME}
  EXECUTABLE_ONLY
  TARGET_LIBRARIES
    ${ITK_LIBRARIES} ITKIOMeta
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)