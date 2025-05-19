import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(CatBreedClassifierApp());
}

class CatBreedClassifierApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cat Breed Classifier',
      home: ClassifierPage(),
    );
  }
}

class ClassifierPage extends StatefulWidget {
  @override
  _ClassifierPageState createState() => _ClassifierPageState();
}

class _ClassifierPageState extends State<ClassifierPage> {
  File? _image;
  String _result = '';
  bool _isLoading = false;

  final picker = ImagePicker();

  Future pickImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile == null) return;

    setState(() {
      _image = File(pickedFile.path);
      _result = '';
    });

    await classifyImage(_image!);
  }

  Future classifyImage(File imageFile) async {
    setState(() {
      _isLoading = true;
    });

    // ✅ Use your Ngrok URL + /predict endpoint
    final uri = Uri.parse("https://e9af-2400-adc7-2111-3e00-70f6-8165-33d9-8e9d.ngrok-free.app/predict");

    var request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));

    try {
      var response = await request.send();
      var responseBody = await response.stream.bytesToString();
      print("Response code: ${response.statusCode}");
      print("Response body: $responseBody");

      if (response.statusCode == 200) {
        var data = json.decode(responseBody);

        setState(() {
          _result = "${data['predicted_class']} (${(data['confidence'] * 100).toStringAsFixed(2)}%)";
        });
      } else {
        setState(() {
          _result = 'Prediction failed: ${response.statusCode}\n$responseBody';
        });
      }
    } catch (e) {
      setState(() {
        _result = 'Error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Cat Breed Classifier')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              _image != null
                  ? Image.file(_image!, height: 200)
                  : Placeholder(fallbackHeight: 200),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: pickImage,
                child: Text('Pick Image'),
              ),
              SizedBox(height: 20),
              _isLoading
                  ? CircularProgressIndicator()
                  : Text(
                      _result,
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 18),
                    ),
            ],
          ),
        ),
      ),
    );
  }
}
