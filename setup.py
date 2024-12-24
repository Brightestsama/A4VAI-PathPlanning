from setuptools import find_packages, setup

package_name = 'pathplanning'

setup(
    name=package_name,
    version='sac-v2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kestrel, Inha University Aerospace Control and Systems Laboratory',
    maintainer_email='kestrel@inha.edu',
    description='Aware4 - Vehicle Awareness Intelligence: Path Planning Algorithm',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'Plan2WP = pathplanning.Plan2WP:main',
        ],
    },
)
