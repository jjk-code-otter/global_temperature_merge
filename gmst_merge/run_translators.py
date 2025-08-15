#  Global Temperature Merge - a package for merging global temperature datasets.
#  Copyright \(c\) 2025 John Kennedy
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path
import glob

def func_launcher(module_name, func_name):
    module = __import__(module_name, fromlist=[None])
    if hasattr(module, func_name):
        print(f"Running {func_name} in {module_name}")
        chosen_fn = getattr(module, func_name)
        chosen_fn()

run_long_conversions = False

for file in glob.glob("translators/*.py"):

    # Get the module name from the filename for import
    file = Path(file).stem
    # And then stick it back onto translators with a dot
    module_name = f'translators.{file}'

    # Run the conversion function if it exists in the module
    func_launcher(module_name, 'convert_file')
    if run_long_conversions:
        func_launcher(module_name, 'convert_file_long')
