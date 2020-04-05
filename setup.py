import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    packages=setuptools.find_packages(),
    name="seirsplus",
    version='0.0.18',
    description='Models of SEIRS epidemic dynamics with extensions, including network-structured populations, testing, contact tracing, and social distancing.',
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/ryansmcgee/SEIRS-network-model",
    author='Ryan Seamus McGee',
    author_email='ryansmcgee@gmail.com',
    license='MIT',
    install_requires=['numpy', 'scipy', 'networkx'],
    zip_safe=False)